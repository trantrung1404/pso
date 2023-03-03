import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Đọc 2 tập dữ liệu huấn luyện và kiểm tra
dataset_train = pd.read_csv('NSL-KDD/KDDTrain+.txt')
dataset_test = pd.read_csv('NSL-KDD/KDDTest+.txt')

print(dataset_train.head())
print(dataset_test.head())

# Thông tin về kích thước của hai tập dữ liệu
print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

# Danh sách tên các cột
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

# Gán tên thuộc tính cho dataset
dataset_train = pd.read_csv('C:/Users/tqt/Downloads/NSL-KDD/KDDTrain+.txt', header=None, names=col_names)
dataset_test = pd.read_csv('C:/Users/tqt/Downloads/NSL-KDD/KDDTest+.txt', header=None, names=col_names)

print(dataset_train.head())
print(dataset_test.head())


# Loại bỏ thuộc tính 'difficulty_level'
dataset_train.drop(['difficulty_level'], axis=1, inplace=True)
dataset_test.drop(['difficulty_level'], axis=1, inplace=True)
print(dataset_train.shape)
print(dataset_train.shape)

# Mô tả thống kê của các tập dữ liệu
print(dataset_train.describe())
print(dataset_test.describe())

# number of attack labels
print(dataset_train['label'].value_counts())
print(dataset_test['label'].value_counts())


# Thay đổi attack labels tương ứng sang loại hình tấn công tương ứng
def change_label(df):
    df.label.replace(['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                      'udpstorm', 'worm'], 'Dos', inplace=True)
    df.label.replace(['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                      'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'], 'R2L',
                     inplace=True)
    df.label.replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe', inplace=True)
    df.label.replace(['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'U2R',
                     inplace=True)


# calling change_label() function
change_label(dataset_train)
change_label(dataset_test)
# distribution of attack classes
print(dataset_train.label.value_counts())
print(dataset_test.label.value_counts())

# DATA NORMALIZATION

# Chọn các cột dữ liệu là số từ tap huấn luyện
numeric_col = dataset_train.select_dtypes(include='number').columns

# Sư dụng standardscaler
std_scaler = StandardScaler()


# Chuan hoa tung cot dư lieu i trong df
# Trich xuat ra array, sau do chuan hoa và tra ve df
def normalization(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df


# calling the normalization() function
data_train = normalization(dataset_train.copy(), numeric_col)
data_test = normalization(dataset_test.copy(), numeric_col)

# data after normalization
print(data_train.head())
print(data_test.head())

# Mã hóa ONE-HOT
# lua chon thuoc tinh dang catergorical
cat_col = ['protocol_type', 'service', 'flag']
# Tao dataframe
categorical = data_train[cat_col]
print(categorical.head())

# Su dung pandas.get_dummies()
categorical = pd.get_dummies(categorical, columns=cat_col)
print(categorical.head())

# BINARY CLASSIFICATION
# changing attack labels into two categories 'normal' and 'abnormal'
bin_label = pd.DataFrame(data_train.label.map(lambda x: 'normal' if x == 'normal' else 'abnormal'))
# creating a dataframe with binary labels (normal,abnormal)
bin_data = data_train.copy()
bin_data['label'] = bin_label

# label encoding (0,1) binary labels (abnormal,normal)
le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['intrusion'] = enc_label

np.save("H:/Trung/AI/AutoEncoder/PSO/labels/le1_classes.npy", le1.classes_, allow_pickle=True)
# dataset with binary labels and label encoded column
print(bin_data.head())

# one-hot-encoding attack label
bin_data = pd.get_dummies(bin_data, columns=['label'], prefix="", prefix_sep="")
bin_data['label'] = bin_label
print(bin_data)

# Biểu đồ phân bố các label bình thường và bất thường
plt.figure(figsize=(8, 8))
plt.pie(bin_data.label.value_counts(), labels=bin_data.label.unique(), autopct='%0.2f%%')
plt.title("Pie chart distribution of normal and abnormal labels")
plt.legend()
plt.savefig('H:/Trung/AI/AutoEncoder/PSO/plots/Pie_chart_binary.png')
plt.show()

# Multi-class Classification
# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = data_train.copy()
multi_label = pd.DataFrame(multi_data.label)

# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['intrusion'] = enc_label
np.save("H:/Trung/AI/AutoEncoder/PSO/labels/le2_classes.npy", le2.classes_, allow_pickle=True)

# one-hot-encoding attack label
multi_data = pd.get_dummies(multi_data, columns=['label'], prefix="", prefix_sep="")
multi_data['label'] = multi_label
print(multi_data)

# pie chart distribution of multi-class labels
plt.figure(figsize=(8, 8))
plt.pie(multi_data.label.value_counts(), labels=multi_data.label.unique(), autopct='%0.2f%%')
plt.title('Pie chart distribution of multi-class labels')
plt.legend()
plt.savefig('H:/Trung/AI/AutoEncoder/PSO/plots/Pie_chart_multi.png')
plt.show()

# Feature Extraction
# creating a dataframe with only numeric attributes of binary class dataset and encoded label attribute
numeric_bin = bin_data[numeric_col]
numeric_bin['intrusion'] = bin_data['intrusion']
# finding the attributes which have more than 0.5 correlation with encoded attack label attribute
corr = numeric_bin.corr()
corr_y = abs(corr['intrusion'])
highest_corr = corr_y[corr_y > 0.5]
highest_corr.sort_values(ascending=True)

# selecting attributes found by using pearson correlation coefficient
numeric_bin = bin_data[['count', 'srv_serror_rate', 'serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                        'logged_in', 'dst_host_same_srv_rate', 'dst_host_srv_count', 'same_srv_rate']]

# joining the selected attribute with the one-hot-encoded categorical dataframe
numeric_bin = numeric_bin.join(categorical)
# then joining encoded, one-hot-encoded, and original attack label attribute
bin_data = numeric_bin.join(bin_data[['intrusion', 'abnormal', 'normal', 'label']])

# saving final dataset to disk
bin_data.to_csv("H:/Trung/AI/AutoEncoder/PSO/datasets/bin_data.csv")
# final dataset for binary classification
print(bin_data)

# creating a dataframe with only numeric attributes of multi-class dataset and encoded label attribute
numeric_multi = multi_data[numeric_col]
numeric_multi['intrusion'] = multi_data['intrusion']

# finding the attributes which have more than 0.5 correlation with encoded attack label attribute
corr = numeric_multi.corr()
corr_y = abs(corr['intrusion'])
highest_corr = corr_y[corr_y > 0.5]
highest_corr.sort_values(ascending=True)

# selecting attributes found by using pearson correlation coefficient
numeric_multi = multi_data[['count', 'logged_in', 'srv_serror_rate', 'serror_rate', 'dst_host_serror_rate',
                            'dst_host_same_srv_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_count',
                            'same_srv_rate']]

# joining the selected attribute with the one-hot-encoded categorical dataframe
numeric_multi = numeric_multi.join(categorical)
# then joining encoded, one-hot-encoded, and original attack label attribute
multi_data = numeric_multi.join(multi_data[['intrusion', 'Dos', 'Probe', 'R2L', 'U2R', 'normal', 'label']])

# saving final dataset to disk
multi_data.to_csv('./datasets/multi_data.csv')

# final dataset for multi-class classification
print(multi_data)
