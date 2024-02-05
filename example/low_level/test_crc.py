from crc_module import get_crc

from unitree_go.msg import LowCmd

lowcmd = LowCmd()
lowcmd.head = [20, 30]
get_crc(lowcmd)