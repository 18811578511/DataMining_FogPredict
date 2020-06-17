# -*- coding: utf-8 -*-
import urllib2
import socket
import time
from bs4 import BeautifulSoup
import threading
from Queue import Queue
import sys
reload(sys)
sys.setdefaultencoding("utf-8")  # 转换格式

sleep_time = 10  # 每执行一段时间后进行休眠的时间
socket.setdefaulttimeout(60)  # 60 秒钟后超时
queue = Queue()  # queue是任务队列
lock = threading.Lock()  # 锁，对多个线程进行同步，保证数据的正确性
lock_file = threading.Lock()  # 创建锁
requireCount = 0  # 执行的城市个数
num_thread = 15  # 开启的线程数


class AQISpider:
    def __init__(self):
        self.cities = self.getCities()
        self.file = open('AQI.txt', 'a')

    # 下载网页源码
    def getUrlRespHtml(self, url):
        heads = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                 'Accept-Charset': 'GB2312,utf-8;q=0.7,*;q=0.7',
                 'Accept-Language': 'zh-cn,zh;q=0.5',
                 'Cache-Control': 'max-age=0',
                 'Connection': 'keep-alive',
                 'Host': 'John',
                 'Keep-Alive': '115',
                 'Referer': url,
                 'User-Agent': 'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.14) Gecko/20110221 Ubuntu/10.10 (maverick) Firefox/3.6.14'}

        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
        urllib2.install_opener(opener)
        req = urllib2.Request(url)
        opener.addheaders = heads.items()
        # 下载失败后重试的次数
        max_try_num = 10
        for tries in range(max_try_num):
            try:
                resp_html = opener.open(req).read()
                break
            except urllib2.HTTPError, e:
                if tries < (max_try_num - 1):
                    continue
                else:
                    print "Has tried %d times to access url %s, all failed!", max_try_num, url
                    print e.code
                    break
        return resp_html

    # 获得城市的具体名字
    def getCities(self):
        site = 'http://pm25.in'
        html = self.getUrlRespHtml(site)
        soup = BeautifulSoup(html, 'html.parser')
        hrefs = soup.find("div", {"class": "all"}).find_all('a')
        cs = []
        for href in hrefs:
            c = href['href']
            c = c[1:len(c)]  # 什么意思????
            cs.append(c)
        return list(set(cs))  # cs中是beijing  shanghai这些

    # 获取当前时间的城市各个监测点的数据
    def getCurrentHour(self):
        # 输出城市总个数
        print 'city count is：' + str(len(self.cities))
        multi = MultiThread(num_thread, len(self.cities))
        # 开启多线程
        multi.do(self.working)
        # 关闭文件
        self.file.close()
        print u'数据下载完毕'

    def working(self):  # working调用了五次（多线程的个数）
        while not queue.empty():  # 这里一共循环了375次（城市个数）
            arguments = queue.get()
            self.do_something_using(arguments)
            queue.task_done()  # 主要是给join用的，每次get后需要调用task_done，直到所有任务都task_done，join才取消阻塞

    # 抓取每个城市的各个监测点的数据
    def do_something_using(self, ind):
        city = self.cities[ind]
        site = 'http://pm25.in/' + city + '.html'
        print str(ind) + ' ' + site
        html = self.getUrlRespHtml(site)
        soup = BeautifulSoup(html, 'html.parser')
        # 时间
        dt = soup.find('div', {'class': 'live_data_time'}).find('p').string
        dt = dt.split('：')[1]
        # 城市名字
        city_name = soup.find('div', {'class': 'city_name'}).find('h2').string
        # 时间和名字
        time_name = dt + '\t'+city_name
        if lock_file.acquire():  # 在用到共同数据的地方加锁
            # 将时间和城市名字写入文件
           # self.file.write(time_name  + '\n')
            # 找到表格
            table = soup.find('table', {'class', 'table table-striped table-bordered table-condensed'})
            # 找到表格中的所有行
            trs = table.find_all('tr')
            for tr in range(len(trs)):  # 遍历表格的每一行
                if tr != 0:
                    tds = trs[tr].find_all('td')  # 找到每一行的所有单元格
                    for td in range(len(tds)):  # 遍历单元格
                        if   td==0:
                            self.file.write(time_name + str(tds[td].get_text()) + '\t')  # 将单元格中的数据写入文件中
                        elif td==10:
                            self.file.write(str(tds[td].get_text()) + '\n')#是1的时候写入到s
                        else:
                            self.file.write(str(tds[td].get_text()) + '\t')  # 是1的时候写入到s
                    #self.file.write('\n')
            lock_file.release()
        # self.file.flush()   # 刷新缓冲区，将缓冲区的数据写入文件
        # 每执行50次进入休眠
        global requireCount  # 记录执行的城市的个数
        if lock.acquire():
            requireCount += 1
            print 'requireCount---' + str(requireCount)
            if requireCount % 50 == 0:
                print 'sleep %s s' % sleep_time
                time.sleep(sleep_time)
            lock.release()


class MultiThread:
    def __init__(self, num, jobs):
        # NUM是并发线程总数
        # JOBS是有多少任务，传进来的jobs是城市个数
        self.NUM = num
        self.JOBS = jobs

    def do(self, working):
        # 把JOBS排入队列
        for i in range(self.JOBS):
            queue.put(i)
        # NUM个线程等待队列
        for i in range(self.NUM):
            t = threading.Thread(target=working)
            t.setDaemon(True)
            t.start()
        # 等待所有JOBS完成
        queue.join()  # q.join()，等到队列为空，再执行别的操作


if __name__ == '__main__':
    while True:
        t1 = time.time()
        spider = AQISpider()
        spider.getCurrentHour()
        t2 = time.time()
        print '单次下载用时%s s' % (t2 - t1)
        time.sleep(3600 - t2 + t1)  # 一个小时后继续抓取

