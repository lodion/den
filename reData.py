import random

import numpy as np
import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy import stats


class UserModel:

    def __init__(self):
        self.Encoder = OneHotEncoder()
        # 指定用户总体数量
        self.SampleNumber = config.SampleNumber
        # 窗口内单次活动时长
        self.DurationSingleActivity = np.random.normal(1, .1) * config.DurationSingleActivityK * np.abs(
            np.random.normal(0, config. \
                             DurationSingleActivityVar * np.abs(np.random.normal(1, .1)),
                             config.SampleNumber))
        # 窗口活动间隔时长
        self.WindowActivityDuration = np.random.normal(1, .1) * 0.7 * np.abs(np.random.normal(
            np.random.uniform(1, 5), np.random.normal(1, .1) * 1.5, config.SampleNumber))
        # 窗体内活动持续时长
        self.FormActivityDuration = np.random.normal(1, .05) * 0.26 * np.square(
            np.random.normal(1, np.random.normal(1, .07) * 2.5, config.SampleNumber))
        # 窗体活动间隔次数 ----------------------[]k
        # self.ActivityInterval = self.Encoder.fit_transform(
        #     np.random.poisson(np.random.normal(1,.1)*config.LambdaParam,config.SampleNumber).reshape(-1,1)).toarray().T
        self.__ActivityInterval__ = None
        self.ActivityInterval = np.random.poisson(np.random.normal(1, .1) * config.LambdaParam, config.SampleNumber)
        # 鼠标键盘活动时长
        self.InputDeviceDuration = np.abs(
            np.random.normal(np.random.uniform(100, 1000), np.random.normal(1, .1) * 3, config.SampleNumber))
        # 窗口名 ----------------------------[]k
        self.__FormNameTag__ = None
        self.FormNameTag = np.floor(7 * np.random.exponential(1 / config.ExponetialRate, config.SampleNumber)).astype(
            int)
        # 活跃窗口数量 -------------------------[]k
        self.__ActiveFormNumber__ = None
        self.ActiveFormNumber = np.floor(np.random.normal(1, .1) * 3 * np.random.exponential(
            np.random.normal(.95, .13) / config.ExponetialRate, config.SampleNumber)).astype(int)
        # 最大化最小化窗口数量 ----------------------[]k
        self.__Max2MinWindowNumber__ = None
        self.Max2MinWindowNumber = np.floor(np.random.normal(1, .1) * 5 * np.random.exponential(
            np.random.normal(.99, .5) / config.ExponetialRate, config.SampleNumber)).astype(int)
        # 窗口自定义情况
        self.p = np.random.random()
        self.SelfSetting = [1 if i > self.p else 0 for i in np.random.random(config.SampleNumber)]
        # 软件切换顺序
        pass
        # 窗口切换耗时
        pass
        # 窗口打开的星期和时间
        pass
        self.Map = None


    def dsac(self):
        plt.hist(self.DurationSingleActivity, bins=30, density=True)
        plt.title('Normal Distribution')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.show()

    def EachOneHotCoding(self, *anothers):
        MulActivityInterval = self.ActivityInterval
        MulFormNameTag = self.FormNameTag
        MulActiveFormNumber = self.ActiveFormNumber
        MulMax2MinWindowNumber = self.Max2MinWindowNumber
        for another in anothers:
            MulActivityInterval = np.append(MulActivityInterval, another.ActivityInterval)
            MulFormNameTag = np.append(MulFormNameTag, another.FormNameTag)
            MulActiveFormNumber = np.append(MulActiveFormNumber, another.ActiveFormNumber)
            MulMax2MinWindowNumber = np.append(MulMax2MinWindowNumber, another.Max2MinWindowNumber)
        MulActivityInterval = self.Encoder.fit_transform(MulActivityInterval.reshape(-1, 1)).toarray()
        MulFormNameTag = self.Encoder.fit_transform(MulFormNameTag.reshape(-1, 1)).toarray()
        MulActiveFormNumber = self.Encoder.fit_transform(MulActiveFormNumber.reshape(-1, 1)).toarray()
        MulMax2MinWindowNumber = self.Encoder.fit_transform(MulMax2MinWindowNumber.reshape(-1, 1)).toarray()
        self.__ActivityInterval__ = MulActivityInterval[:config.SampleNumber]
        self.__FormNameTag__ = MulFormNameTag[:config.SampleNumber]
        self.__ActiveFormNumber__ = MulActiveFormNumber[:config.SampleNumber]
        self.__Max2MinWindowNumber__ = MulMax2MinWindowNumber[:config.SampleNumber]
        self.Map = np.concatenate((
            np.array([self.DurationSingleActivity]),
            np.array([self.WindowActivityDuration]),
            np.array([self.FormActivityDuration]),
            self.__ActivityInterval__.T,
            np.array([self.InputDeviceDuration]),
            self.__FormNameTag__.T,
            self.__ActiveFormNumber__.T,
            self.__Max2MinWindowNumber__.T,
            np.array([self.SelfSetting])
        ), axis=0).T
        self.Map = self.Map / np.linalg.norm(self.Map, axis=0)[:, np.newaxis].reshape(-1)
        for i, another in enumerate(anothers):
            another.__ActivityInterval__ = MulActivityInterval[
                                           (i + 1) * config.SampleNumber:(i + 2) * config.SampleNumber]
            another.__FormNameTag__ = MulFormNameTag[(i + 1) * config.SampleNumber:(i + 2) * config.SampleNumber]
            another.__ActiveFormNumber__ = MulActiveFormNumber[
                                           (i + 1) * config.SampleNumber:(i + 2) * config.SampleNumber]
            another.__Max2MinWindowNumber__ = MulMax2MinWindowNumber[
                                              (i + 1) * config.SampleNumber:(i + 2) * config.SampleNumber]
            another.Map = np.concatenate((
                np.array([another.DurationSingleActivity]),
                np.array([another.WindowActivityDuration]),
                np.array([another.FormActivityDuration]),
                another.__ActivityInterval__.T,
                np.array([another.InputDeviceDuration]),
                another.__FormNameTag__.T,
                another.__ActiveFormNumber__.T,
                another.__Max2MinWindowNumber__.T,
                np.array([another.SelfSetting])
            ), axis=0).T
            another.Map = another.Map / np.linalg.norm(another.Map, axis=0)[:, np.newaxis].reshape(-1)

    def TransformKS(self, another):
        if self.Map is None or another.Map is None:
            self.EachOneHotCoding(another)
        sampleUserFirst = np.array([np.sum(random.sample(list(self.Map), config.NNN), axis=0)])
        sampleUserSecond = np.array([np.sum(random.sample(list(another.Map), config.NNN), axis=0)])
        for i in range(config.NUM - 1):
            # print(np.sum(random.sample(list(self.Map),config.NNN),axis=0).shape)
            # print(sampleUserSecond.shape)
            sampleUserFirst = np.vstack((sampleUserFirst, np.sum(random.sample(list(self.Map), config.NNN), axis=0)))
            sampleUserSecond = np.vstack(
                (sampleUserSecond, np.sum(random.sample(list(another.Map), config.NNN), axis=0)))
        sampleUserSecond = sampleUserSecond.T
        sampleUserFirst = sampleUserFirst.T
        pLs = []
        count = 0
        for i, j in zip(sampleUserFirst, sampleUserSecond):
            _, p_value = stats.ks_2samp(i, j)
            # print(p_value)
            pLs.append(p_value)
            count = count + 1 if p_value < config.DenyDomain else count
        return pLs, count / len(pLs)


um = UserModel()
# # print(um.ActivityInterval)
# # print(um.Map)
# # np.setx('umMap.csv',um.Map)
print(um.TransformKS(UserModel()))
print(um.TransformKS(um))
