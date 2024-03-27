from crc_module import get_crc

from unitree_go.msg import LowCmd

lowcmd = LowCmd()
lowcmd.head = [20, 30]
for k in range(20):
    lowcmd.motor_cmd[k].mode = 1
    lowcmd.motor_cmd[k].q = float(k)
    lowcmd.motor_cmd[k].dq = float(k)
    lowcmd.motor_cmd[k].tau = float(k)
    lowcmd.motor_cmd[k].kp = float(k)
    lowcmd.motor_cmd[k].kd = float(k)


lowcmd.led = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print(get_crc(lowcmd))
lowcmd.crc = get_crc(lowcmd)