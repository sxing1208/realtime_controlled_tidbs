from PyQt5.QtWidgets import QApplication, QMessageBox,QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QMessageBox, QPlainTextEdit
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
from PyQt5.QtCore import Qt

import numpy as np
import sys
import os
import asyncio
import bleak
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from tensorflow.keras.models import load_model

_datetime = datetime.now().strftime("%Y_%m_%d_%H%M%S")
open(f"datafiles\{_datetime}.csv","w")

stimulation_param_dev = "MYBLE"
tremor_detection_dev = "Arduino"

stimulation_service_id = '0000180c-0000-1000-8000-00805f9b34fb'
tremor_service_id = '0000180c-0000-1000-8000-00805f9b34fb'

stimulation_char_id_1 = '00002a6f-0000-1000-8000-00805f9b34fb'
stimulation_char_id_2 = '00002a70-0000-1000-8000-00805f9b34fb'
tremor_char_id = '00002a6e-0000-1000-8000-00805f9b34fb'

def calculate_window(scale_width=0.5, scale_height=0.5):
    screen = QApplication.primaryScreen().geometry()
    screen_width, screen_height = screen.width(), screen.height()
    window_width, window_height = int(screen_width * scale_width), int(screen_height * scale_height)
    x_pos, y_pos = (screen_width - window_width) // 2, (screen_height - window_height) // 2
    return window_width, window_height, x_pos, y_pos

class BLEHandler(QObject):
    message = pyqtSignal(str)
    incoming_data = pyqtSignal(list)
    error = pyqtSignal(str)
    ready = pyqtSignal()
    connected = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.main())
        self.loop.run_forever()

    async def main(self):
        self.m_scanner = bleak.BleakScanner()
        self.m_client_stimulation = None
        self.m_client_tremor = None
        self.isDeviceDiscovered = False

        self.devicesDict = {}

        self.stimulationServicesDict = {}
        self.tremorServicesDict = {}
        self.stimulationCharDict = {}
        self.tremorCharDict = {}

        self.plotthread = QThread()
        
        await self.scanDevices()

    async def scanDevices(self):
        self.message.emit("Scanning")
        devices = await bleak.BleakScanner.discover()
        for device in devices:
            self.devicesDict[device.name] = device

        try:
            self.m_client_stimulation = bleak.BleakClient(self.devicesDict[stimulation_param_dev])
            self.m_client_tremor = bleak.BleakClient(self.devicesDict[tremor_detection_dev])
        except:
            self.error.emit("unable to connect")

        self.message.emit("Searching for services")

        await self.scanServices()

    async def scanServices(self):
        try:
            await self.m_client_stimulation.connect()
            await self.m_client_tremor.connect()
        except:
            self.error.emit("unable to connect")
        for service in self.m_client_stimulation.services:
            self.stimulationServicesDict[service.uuid] = service
        for service in self.m_client_tremor.services:
            self.tremorServicesDict[service.uuid] = service

        self.message.emit("Searching for characteristics")

        self.stimulationService = self.stimulationServicesDict[stimulation_service_id]
        self.tremorService = self.tremorServicesDict[tremor_service_id]
        stimulation_chars = self.stimulationService.characteristics
        tremor_chars = self.tremorService.characteristics
        for char in stimulation_chars:
            self.stimulationCharDict[char.uuid] = char
        for char in tremor_chars:
            self.tremorCharDict[char.uuid] = char

        self.stimulationChar_1 = self.stimulationCharDict[stimulation_char_id_1]
        self.stimulationChar_2 = self.stimulationCharDict[stimulation_char_id_2]
        self.tremorChar = self.tremorCharDict[tremor_char_id]

        try:
            await self.m_client_tremor.start_notify(self.tremorChar, self.decodeRoutine)
            self.ready.emit()
            self.connected.emit([self.m_client_stimulation,self.stimulationChar_1,self.stimulationChar_2])
        except:
            self.error.emit("unable to start notification")

    def decodeRoutine(self, char, value):
        try:
            decoded_value = value.decode("UTF-8")
            decoded_list = decoded_value.split(",")
        except:
            self.error.emit("unable to decode")
            self.close()
        
        data_list = []

        for i in range(len(decoded_list)):
            try:
                data_list.append(int(decoded_list[i].replace('\x00','')))
            except:
                self.error.emit("unable to decode")
        if len(data_list) == 3:
            self.incoming_data.emit(data_list)
        else:
            self.error.emit("datalength_inccorect")

class BLEWriter(QObject):
    message = pyqtSignal(str)

    def run(self):
        super().__init__()
        self.loop = None
        self.m_client_stimulation = None
        self.stimulationChar_1 = None
        self.stimulationChar_2 = None

    def onConnected(self, received_list):
        self.m_client_stimulation = received_list[0]
        self.stimulationChar_1 = received_list[1]
        self.stimulationChar_2 = received_list[2]

    def onWriteFreq(self, freq_list):
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.write_to_char(freq_list))
    
    async def write_to_char(self, freq_list):
        await self.m_client_stimulation.write_gatt_char(stimulation_char_id_1, freq_list[0].to_bytes(2, byteorder="little"))
        await self.m_client_stimulation.write_gatt_char(stimulation_char_id_2, freq_list[1].to_bytes(2, byteorder="little"))
        self.message.emit(f"Data written to characteristic")

class Saver(QObject):
    saved = pyqtSignal()

    def run(self):
        super().__init__()

    def save_to_csv(self, data):
        with open(f"datafiles/{_datetime}.csv","a") as f:
            f.write(str(data)+'\n')

class RealTimeML(QObject):
    predictormessage = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.model = None

    def run(self):
        self.model = load_model("my_model_updates.h5")

    def on_new_data(self, data):
        data = np.array(data)/100
        data = np.expand_dims(data.T, axis=0)
        data = np.expand_dims(np.sqrt(np.sum(data**2,axis=-1)),axis=-1)
        results = self.model.predict(data)[0]
        self.predictormessage.emit(str(results*100))

class VisualizationWindow(QWidget):
    savedata = pyqtSignal(int)
    writefreq = pyqtSignal(list)
    predictdata = pyqtSignal(list)
    first_data_pass = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        self.dataframe = [deque(maxlen=20),deque(maxlen=20),deque(maxlen=20)]
        self.isFirstPass = True

        self.thread1 = QThread()
        self.thread2 = QThread()
        self.thread3 = QThread()
        self.thread4 = QThread()
        self.worker = BLEHandler()
        self.saver = Saver()
        self.writer = BLEWriter()
        self.predictor = RealTimeML()

        self.worker.moveToThread(self.thread1)
        self.saver.moveToThread(self.thread2)
        self.writer.moveToThread(self.thread3)
        self.predictor.moveToThread(self.thread4)

        self.thread1.started.connect(self.worker.run)
        self.thread2.started.connect(self.saver.run)
        self.thread3.started.connect(self.saver.run)
        self.thread4.started.connect(self.predictor.run)
        self.worker.message.connect(self.onMessage)
        self.worker.error.connect(self.onError)
        self.worker.ready.connect(self.onReady)
        self.worker.incoming_data.connect(self.onNewData)
        self.savedata.connect(self.saver.save_to_csv)
        self.writefreq.connect(self.writer.onWriteFreq)
        self.writer.message.connect(self.onMessage)
        self.worker.connected.connect(self.writer.onConnected)
        self.first_data_pass.connect(self.predictor.on_new_data)
        self.predictdata.connect(self.predictor.on_new_data)
        self.predictor.predictormessage.connect(self.onPredictionMessage)

        self.arrayidx = 0
        self.m_array = np.array([])

        self.thread1.start()
        self.thread2.start()
        self.thread3.start()
        self.thread4.start()

        self.initUI()

    def plotUpdate(self, frame):
        self._line0.set_xdata(range(len(self.dataframe[0])))
        self._line0.set_ydata(self.dataframe[0])
        self._line1.set_xdata(range(len(self.dataframe[1])))
        self._line1.set_ydata(self.dataframe[1])
        self._line2.set_xdata(range(len(self.dataframe[2])))
        self._line2.set_ydata(self.dataframe[2])

        self._ax[0].relim()
        self._ax[1].relim()
        self._ax[2].relim()
        self._ax[0].autoscale_view()
        self._ax[1].autoscale_view()
        self._ax[2].autoscale_view()

        return [self._line0,self._line1,self._line2]

    @pyqtSlot()
    def _plot(self):
        self._animation = FuncAnimation(self._fig, self.plotUpdate, interval=10, cache_frame_data=False, blit=True)
        self._canvas.draw_idle()

    def initUI(self):
        self.setWindowTitle('TIDBS_Client')
        window_width, window_height, x_pos, y_pos = calculate_window(scale_width=0.8, scale_height=0.8)
        self.setGeometry(x_pos, y_pos, window_width, window_height)

        self.textfield = QPlainTextEdit(self)
        self.textfield.setEnabled(False)
        self.statuslabel = QLabel(self)
        self.statuslabel.setText("Attempting to Connect")
        self.predictorlabel = QLabel(self)
        self.predictorlabel.setText("Waiting for Model to start")
        self.labelchar1 = QLabel(self)
        self.labelchar1.setText("Freqeuncy 1 (kHz)")
        self.labelchar2 = QLabel(self)
        self.labelchar2.setText("Freqeuncy 2 (kHz)")
        self.textchar1 = QPlainTextEdit(self)
        self.textchar2 = QPlainTextEdit(self)
        self.writebutton = QPushButton("Update Stimulation Freqeuncy",self)
        self.writebutton.clicked.connect(self.onWriteFreqButton)
        self.writebutton.setEnabled(False)

        self._fig, self._ax = plt.subplots(3, 1, sharex=True, sharey=True)
        self._line0, = self._ax[0].plot(self.dataframe[0])
        self._line1, = self._ax[1].plot(self.dataframe[1])
        self._line2, = self._ax[2].plot(self.dataframe[2])
        self._title = "ADC"
        self._xlabel = "Time (a.u.)"
        self._ylabel = "Value (a.u.)"
        self._canvas = FigureCanvas(self._fig)

        self._ax[0].set_title(self._title)
        self._ax[2].set_xlabel(self._xlabel)
        self._ax[1].set_ylabel(self._ylabel)
            
        self._ax[0].tick_params(labelleft=False)
        self._ax[1].tick_params(labelleft=False)
        self._ax[2].tick_params(labelleft=False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.labelchar1)
        layout.addWidget(self.textchar1)
        layout.addWidget(self.labelchar2)
        layout.addWidget(self.textchar2)
        layout.addWidget(self.writebutton)
        layout.addWidget(self.statuslabel)
        layout.addWidget(self.predictorlabel)
        layout.addWidget(self.textfield)
        layout.addWidget(self._canvas)

    def onError(self,message):
        QMessageBox.warning(self,"Error",message)

    def onNewData(self,data):
        self.textfield.setPlainText(str(data))
        self.dataframe[0].append(data[0])
        self.dataframe[1].append(data[1])
        self.dataframe[2].append(data[2])
        self.savedata.emit(data)
        if self.isFirstPass and len(self.dataframe[0]) == 20:
            self.first_data_pass.emit(self.dataframe)
            self.isFirstPass = False

    def onMessage(self,message):
        self.statuslabel.setText(message)

    def onPredictionMessage(self,message):
        self.predictorlabel.setText(message)
        self.predictdata.emit(self.dataframe)

    def onReady(self):
        self.writebutton.setEnabled(True)
        self.statuslabel.setText("Ready")
        self._plot()

    def onWriteFreqButton(self):
        try:
            char1 = int(float(self.textchar1.toPlainText())*10)
            char2= int(float(self.textchar2.toPlainText())*10)
            self.writefreq.emit([char1,char2])
        except:
            QMessageBox.warning(self,"Error", "Invalid Frequency Entry")


def main():
    app = QApplication(sys.argv)
    ex = VisualizationWindow()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()