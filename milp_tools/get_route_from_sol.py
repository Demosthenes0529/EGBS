#!/usr/bin/env python3
import re
import sys

def parse_sol_file(filename, blocksize=32):
    """
    解析SPECK差分路径的sol文件
    blocksize: SPECK的分组大小（32, 48, 64, 96, 128等）
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # 提取目标值
    obj_match = re.search(r'Objective value = ([\d\.eE+-]+)', content)
    obj_value = float(obj_match.group(1)) if obj_match else None
    
    # 提取输入差分比特 (p0 到 p_{blocksize-1})
    input_diff = {}
    for i in range(blocksize):
        match = re.search(r'^p%d\s+([\d\.eE+-]+)' % i, content, re.MULTILINE)
        if match:
            val = float(match.group(1))
            # 判断是否为1（考虑浮点精度）
            input_diff[i] = 1 if val > 0.5 else 0
        else:
            input_diff[i] = 0
    
    # 构建输入差分的二进制字符串和十六进制
    input_bits = ''.join(str(input_diff[i]) for i in range(blocksize))
    # 计算十六进制表示的位数（每个十六进制字符代表4位）
    hex_digits = (blocksize + 3) // 4
    input_hex = format(int(input_bits, 2), f'0{hex_digits}X')
    
    # 提取每一轮的差分
    rounds_diff = []
    
    # 首先找出有多少轮
    max_round = 0
    for line in content.split('\n'):
        match = re.match(r'^p\d+Rd(\d+)', line)
        if match:
            rnd = int(match.group(1))
            max_round = max(max_round, rnd)
    
    print(f"Found {max_round} rounds of differentials")
    
    for rnd in range(1, max_round + 1):
        round_bits = {}
        # 初始化所有比特为0
        for i in range(blocksize):
            round_bits[i] = 0
        
        # 提取这一轮所有非零的差分
        for i in range(blocksize):
            pattern = r'^p%dRd%d\s+([\d\.eE+-]+)' % (i, rnd)
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                val = float(match.group(1))
                # 判断是否为1（考虑浮点精度）
                if val > 0.5:
                    round_bits[i] = 1
        
        # 构建二进制字符串
        bits = ''.join(str(round_bits[i]) for i in range(blocksize))
        hex_val = format(int(bits, 2), f'0{hex_digits}X')
        rounds_diff.append((bits, hex_val))
    
    return obj_value, input_bits, input_hex, rounds_diff, max_round

def format_binary_string(bits, blocksize):
    """格式化二进制字符串，每8位一组"""
    groups = []
    for i in range(0, blocksize, 8):
        group = bits[i:i+8]
        if group:
            groups.append(group)
    return ' '.join(groups)

def print_detailed_differential(filename, blocksize=32):
    obj, input_bits, input_hex, rounds, num_rounds = parse_sol_file(filename, blocksize)
    
    print("=" * 80)
    print(f"SPECK{blocksize}/{num_rounds} Differential Path Analysis")
    print("=" * 80)
    print(f"Probability: 2^{{{-obj:.6f}}}" if obj else "Probability: N/A")
    print()
    
    # 找出所有非零位置
    non_zero_positions = [i for i, bit in enumerate(input_bits) if bit == '1']
    print(f"Input difference has active bits at positions: {non_zero_positions}")
    print()
    
    print("Round 0 (Plaintext):")
    print(f"  Hex: 0x{input_hex}")
    print(f"  Binary: {format_binary_string(input_bits, blocksize)}")
    print()
    
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        active_positions = [i for i, bit in enumerate(bits) if bit == '1']
        if active_positions:
            print(f"Round {rnd}:")
            print(f"  Hex: 0x{hex_val}")
            print(f"  Binary: {format_binary_string(bits, blocksize)}")
            print(f"  Active bits: {active_positions}")
            print()
        else:
            print(f"Round {rnd}: All zero (no active differences)")
            print()

def print_code_format(filename, blocksize=32):
    obj, input_bits, input_hex, rounds, num_rounds = parse_sol_file(filename, blocksize)
    
    print("=" * 80)
    print("Code Format (Ready to use)")
    print("=" * 80)
    print()
    
    # 确定C/C++的数据类型
    if blocksize <= 32:
        c_type = "uint32_t"
    elif blocksize <= 64:
        c_type = "uint64_t"
    else:
        c_type = f"uint{blocksize}_t"
    
    print("# Python")
    print(f"num_rounds = {num_rounds}")
    print(f"probability_exponent = {obj:.6f}")
    print(f"input_diff = 0x{input_hex}")
    print()
    
    print("round_diffs = [")
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        if int(hex_val, 16) == 0:
            print(f"    0x{hex_val},  # Round {rnd} - no active bits")
        else:
            print(f"    0x{hex_val},  # Round {rnd}")
    print("]")
    print()
    
    print(f"# C/C++ (SPECK{blocksize})")
    print(f"const int NUM_ROUNDS = {num_rounds};")
    print(f"const double PROB_EXP = {obj:.6f};")
    print(f"const {c_type} INPUT_DIFF = 0x{input_hex};")
    print(f"const {c_type} ROUND_DIFFS[{num_rounds}] = {{")
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        if rnd < num_rounds:
            print(f"    0x{hex_val},  // Round {rnd}")
        else:
            print(f"    0x{hex_val}   // Round {rnd}")
    print("};")
    print()
    
    # 输出二进制格式的数组
    print("# Binary format (LSB first) - useful for verification")
    print(f"input_diff_bits = [{', '.join(input_bits)}]")
    print()
    print("# Active positions per round:")
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        active = [i for i, bit in enumerate(bits) if bit == '1']
        if active:
            print(f"Round {rnd}: {active}")
        else:
            print(f"Round {rnd}: none")

def print_compact_path(filename, blocksize=32):
    obj, input_bits, input_hex, rounds, num_rounds = parse_sol_file(filename, blocksize)
    
    print("=" * 80)
    print(f"SPECK{blocksize}/{num_rounds} Differential Trail")
    print("=" * 80)
    print(f"Probability: 2^{{{-obj:.6f}}}")
    print()
    print("Differential path (hex):")
    print(f"  Input:  0x{input_hex}")
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        print(f"  Round {rnd:2d}: 0x{hex_val}")
    print()
    print("Differential path (active bits positions):")
    print(f"  Input:  {[i for i, bit in enumerate(input_bits) if bit == '1']}")
    for rnd, (bits, hex_val) in enumerate(rounds, 1):
        active = [i for i, bit in enumerate(bits) if bit == '1']
        print(f"  Round {rnd:2d}: {active if active else 'none'}")

def main():
    # 可以在这里修改参数
    blocksize = 32  # SPECK的分组大小: 32, 48, 64, 96, 128
    rounds = 5      # 轮数
    rev = 1
    
    # 构建文件名
    if rev == 1:
        filename = f"/home/user/speck_output/speck{blocksize}/speck{blocksize}round{rounds}-rev.sol"
    else:
        filename = f"/home/user/speck_output/speck{blocksize}/speck{blocksize}round{rounds}.sol"
    
    # 检查文件是否存在
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Usage: python script.py [blocksize] [rounds]")
        print("Example: python script.py 32 9")
        print("         python script.py 64 10")
        return
    
    # 解析并显示结果
    print_detailed_differential(filename, blocksize)
    print("\n" + "=" * 80 + "\n")
    print_compact_path(filename, blocksize)
    print("\n" + "=" * 80 + "\n")
    print_code_format(filename, blocksize)

if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        blocksize = int(sys.argv[1])
        if len(sys.argv) > 2:
            rounds = int(sys.argv[2])
    else:
        blocksize = 128
        rounds = 10
    
    rev = 1
    # 构建文件名
    if rev == 1:
        filename = f"/home/user/speck_output/speck{blocksize}/speck{blocksize}round{rounds}-rev.sol"
    else:
        filename = f"/home/user/speck_output/speck{blocksize}/speck{blocksize}round{rounds}.sol"
    
    # 检查文件是否存在
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Usage: python script.py [blocksize] [rounds]")
        print("Example: python script.py 32 9")
        print("         python script.py 64 10")
        sys.exit(1)
    
    # 解析并显示结果
    print_detailed_differential(filename, blocksize)
    print("\n" + "=" * 80 + "\n")
    print_compact_path(filename, blocksize)
    print("\n" + "=" * 80 + "\n")
    print_code_format(filename, blocksize)