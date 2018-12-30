# encoding=utf-8
import ConfigParser

# 读取配置文件
cf = ConfigParser.ConfigParser()
cf.read("./config.ini")
# 输出文本的路径，为防止中文路径乱码，使用UNICODE编码
path_dataroot=cf.get("path", "datapath")
path_dataraw=cf.get("path", "datarawpath")
path_dataseg=cf.get("path", "datasegpath")
path_datagood=cf.get("path", "goodpath")
path_datahighq=cf.get("path", "highqpath")
path_datahighq5w=cf.get("path", "highq5wpath")
path_model = cf.get("path", "modelpath")

raw_ulog_b09=cf.get("file", "raw_ulog_b09")
raw_ulog_d09=cf.get("file", "raw_ulog_d09")
raw_ulog_b18=cf.get("file", "raw_ulog_b18")
raw_ulog_d18=cf.get("file", "raw_ulog_d18")

# dict_path = cf.get("dict", "dict_path")
otherdict_folder=cf.get("dict", "otherdict_folder")
stop_words_path = cf.get("dict", "stop_words_path")
kbase_dict_path = cf.get("dict", "kbase_dict_path")
idf_dict_path = cf.get("dict", "idf_path")
jieba_dict_path = cf.get("dict", "jieba_dict_path")

#mysql conf
mysqlhost = cf.get("mysql","host")
mysqluser = cf.get("mysql","user")
mysqlpasswd = cf.get("mysql","passwd")


