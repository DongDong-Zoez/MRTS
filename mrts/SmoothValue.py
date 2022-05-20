import time
import numpy as np
import os 

class SmoothValue:

    @classmethod
    def setting(cls, file_path='logger.txt'):
        cls.file_path = file_path
        cls.string = ''
        with open(cls.file_path, 'w') as file:
            cls.start_time = time.time()
            text = f'Start at {SmoothValue.timeRecorder()}'
            floor, ceil = cls.getSpace(text=text)
            info = '#' * 60 + '\n' + '#' + ' ' * floor +  text + ' ' * ceil + '#' + '\n' + '#' * 60
            SmoothValue.log(info)
            file.close()

    @staticmethod
    def timeRecorder():
        return time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())

    @classmethod
    def addCallback(cls, text):
        info = f'{text} | pass: {np.round(time.time() - cls.start_time,2)}s | time: {SmoothValue.timeRecorder()}'
        print(info)
        SmoothValue.log(info)

    @staticmethod
    def getSpace(text, lwd=58):
        floor = int(np.floor((lwd -  len(text)) / 2))
        ceil = int(np.ceil((lwd -  len(text)) / 2))
        return floor, ceil

    @classmethod
    def durationTime(cls):
        cls.end_time = time.time()
        cls.duration = cls.end_time - cls.start_time
        hours, minutes, seconds = cls.convert_timedelta(cls.duration)
        text = f'End at {SmoothValue.timeRecorder()} | during time: {hours}h {minutes}min {seconds}s'
        floor, ceil = cls.getSpace(text=text)
        info = '#' * 60 + '\n' + '#' + ' ' * floor + text + ' ' * ceil + '#'
        SmoothValue.log(info)

    @classmethod
    def log(cls, line):

        file = open(cls.file_path, 'a')
        file.write(line)
        file.write('\n')

    @staticmethod
    def convert_timedelta(duration):
        seconds = duration
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = (seconds % 60)
        return str(int(hours)), str(int(minutes)), str(np.round(seconds, 2))

    @classmethod
    def info(cls, *args, **kwargs):
        for key, value in kwargs.items():
            cls.string = cls.string + f'{key}: {value}\n'
        for arg in args:
            cls.string = cls.string + arg + '\n'

    @classmethod
    def writeInfo(cls, *args, **kwargs):
        cls.string = '#' * 60 + '\n' + cls.string
        cls.string = cls.string + f'{cls.success} articles are fetched sucessfully\n'
        cls.string = cls.string + f'{cls.fail} articles are failed due to the articles were deleted.'

        SmoothValue.log(cls.string)

    @classmethod
    def error(cls, text):
        raise Exception(text)