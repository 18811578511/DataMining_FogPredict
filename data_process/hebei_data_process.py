#coding:utf-8
import pandas as pd


def delete_repetitive(read_dir, write_dir):
    """
    去除重复的数据
    :param read_dir:
    :param write_dir:
    """
    # readDir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/data_after1.txt'
    # writeDir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/data_after2.txt'
    lines_seen = set()
    with open(read_dir, 'r', encoding='UTF-8') as f:
        with open(write_dir, "w", encoding='UTF-8') as outfile:
            for line in f:
                if line not in lines_seen:
                    outfile.write(line)
                    lines_seen.add(line)
            outfile.close()


def find_city_data(points, read_dir, write_dir):
    """
    根据每个城市的监测点名称找到对应的数据
    :param points:
    :param read_dir:
    :param write_dir:
    """
    with open(read_dir, 'r', encoding='UTF-8') as inFile:
        with open(write_dir, "w", encoding='UTF-8') as outFile:
            for point in points:
                inFile.seek(0)  # 回到文件开头
                for line in inFile:
                    # 筛选正确的数据
                    if (len(line.split('	')) == 12) and (point == line.split('	')[1]) and ('2019' in line.split('	')[0]) and (len(line.split('	')[0]) == 19):
                        t = 0
                        for each in line.split('	'):
                            if len(each) == 0:  # 不能有空值
                                t = 1
                                break
                        if t == 0:
                            outFile.write(line)
    inFile.close()
    outFile.close()


def complete_data(read_dir, write_dir):
    """
    进行均值补全，并且删除两个时间一样的,将数字转换为float格式，保存为csv文件
    :param read_dir:
    :param write_dir:
    """
    whole_data = []
    # 读取文件并存在list中
    with open(read_dir, 'r', encoding='UTF-8') as inFile:
        for line in inFile:
            line = line.strip('\n')
            one = line.split('	')
            whole_data.append(one)
    inFile.close()
    # 进行均值补全
    for i in range(len(whole_data)):
        for j in range(len(whole_data[i])):
            if j != 0 and j != 1 and j != 3 and j != 4:
                if '_' == whole_data[i][j]:  # 如果当前位置为空值
                    # 如果当前行是最后一行，或者下一刻的值还是空值，就等于上一时刻的值
                    if (i == (len(whole_data)-1)) or ('_' == whole_data[i+1][j]):
                        whole_data[i][j] = float(whole_data[i-1][j])
                    else:   # 如果下一时刻有值，就进行均值补全（上一时刻一定是有值的）
                        whole_data[i][j] = float((float(whole_data[i-1][j])+float(whole_data[i+1][j]))/2)
                else:  # 如果当前行有值
                    whole_data[i][j] = float(whole_data[i][j])  # 转换成float类型
    # 删除两个时间一样的（有问题的）
    num = 0
    a = len(whole_data)
    for i in range(a):
        if (i >= 1) and (i <= (len(whole_data)-1)):
            if whole_data[i][0] == whole_data[i-1][0]:
                num = num + 1
                print('第'+str(num)+'个，请处理：序号为'+str(i))
                print(whole_data[i-1])
                print(whole_data[i])
                stay = input('请选择要删除的数据：')
                if stay == '1':
                    del whole_data[i-1]
                    i = i-1
                else:
                    del whole_data[i]
                    i = i - 1
    # 结果保存为csv文件
    name = ['time', 'point', 'aqi', 'aqi_type', 'first_pollution', 'PM2.5', 'PM10', 'CO', 'NO2', '03_1', '03_8', 'SO2']
    all_data = pd.DataFrame(columns=name, data=whole_data)
    all_data.to_csv(write_dir, encoding='utf_8_sig')


points = ['保定华电二区', '保定接待中心', '保定地表水厂', '保定胶片厂', '保定监测站',
          '廊坊药材公司', '廊坊开发区', '廊坊北华航天学院', '廊坊河北工业大学',
          '张家口人民公园', '张家口五金库', '张家口世纪豪园', '张家口北泵房', '张家口烟厂',
          '承德铁路', '承德中国银行', '承德开发区', '承德文化中心', '承德离宫',
          '唐山供销社', '唐山雷达站', '唐山物资局', '唐山陶瓷公司', '唐山十二中', '唐山小山',
          '石家庄职工医院', '石家庄高新区', '石家庄西北水源', '石家庄西南高教', '石家庄世纪公园', '石家庄人民会堂', '石家庄封龙山', '石家庄22中南校区',
          '沧州沧县城建局', '沧州电视转播站', '沧州市环保局',
          '秦皇岛北戴河环保局', '秦皇岛第一关', '秦皇岛监测站', '秦皇岛建设大厦',
          '衡水电机北厂', '衡水市监测站',
          '邢台邢师高专', '邢台路桥公司', '邢台市环保局',
          '邯郸环保局', '邯郸东污水处理厂', '邯郸矿院', '邯郸丛台公园']
# read_dir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/data_before.txt'
read_dir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/hebei_data_before.txt'
# write_dir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/hebei_data_before.txt'
write_dir = 'C:/Users/xue/Desktop/毕设/数据/data_txt/hebei_before_all.csv'

# delete_repetitive(read_dir,write_dir)
# find_city_data(points, read_dir, write_dir)
complete_data(read_dir, write_dir)
print('over')
# 75185

