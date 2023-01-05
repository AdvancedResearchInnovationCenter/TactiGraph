import pandas as pd
import matplotlib.pyplot as plt
import sys
import paramiko
import numpy as np

def main():
    privatekeyfile = '/home/hussain/.ssh/almesbar_private_key'
    mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.connect(pkey=mykey, hostname='login.almesbar.ku.ac.ae', username='100062332', password='Hasoo0oon1515__')

    ftp_client = ssh_client.open_sftp()
    file = ftp_client.open(f'/l/proj/kuin0034/hussain/results/{sys.argv[1]}/train_log.csv')
    log = pd.read_csv(file)
    print(log)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.sca(axes[0])
    plt.grid()
    plt.plot(log['epoch'], np.degrees(log['train_loss']), label='train')
    plt.plot(log['epoch'], np.degrees(log['val_loss']), label='val')
    plt.legend()

    plt.sca(axes[1])
    plt.grid()
    plt.plot(log['epoch'], np.degrees(log['lr']))

    plt.show()
if __name__ == "__main__":
    main()