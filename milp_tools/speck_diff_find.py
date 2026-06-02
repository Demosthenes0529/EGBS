from functools import reduce
import math
import random
from MILPSbox import *
from random import *
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)




class speck():
    def __init__(self, blocksize):
        self.BlockSize = blocksize

    def genVars_InVars_at_Round(self, r):
        # 生成指定轮次的输入变量名
        assert r >= 1
        if r == 1:
            return ['p'+str(j) for j in range(self.BlockSize)]
        if r > 1 :      
            return ['p'+str(j)+'Rd'+str(r-1) for j in range(self.BlockSize)]
        
    def rotl(self, X, n, r):
        # 循环左移
        assert r >= 1
        temp = [None]*n
        for i in range(n-r):
            temp[i] = X[i+r]
        for i in range(n-r,n):
            temp[i] = X[i-n+r]
        return temp
    
    def rotr(self, X, n, r):
        # 循环右移
        assert r >= 1
        temp = [None]*n
        for i in range(r):
            temp[i] = X[n-r+i]
        for i in range(r,n):
            temp[i] = X[i-r]
        return temp
    
    def genVars_Objective(self, r):
        # 生成目标函数变量，用于最小化活跃S盒数量
        h = self.BlockSize
        n = int(h/2)
        assert r >= 1
        return ['pro'+str(i)+'Rd'+str(r-1) for i in range(n-1)]
    
    def genConstraints_of_Round(self, r):
        assert r>=1
        constraints = list()
        X = self.genVars_InVars_at_Round(r)      # 本轮输入变量
        Y = self.genVars_InVars_at_Round(r+1)    # 下一轮输出变量
        h = self.BlockSize
        n = int(h/2)
        x0 = X[0:n]      # 左半部分
        x1 = X[n:2*n]    # 右半部分
        y0 = Y[0:n]      # 输出左半部分
        y1 = Y[n:2*n]    # 输出右半部分
        if n == 16:
            x2 = self.rotr(x0, 16, 7)
            x3 = self.rotl(x1, 16, 2)
        else:
            x2 = self.rotr(x0, n, 8)
            x3 = self.rotl(x1, n, 3)
            
        constraints += ConstraintGenerator.xorConstraints(x3, y1, y0)    # 实现：y0 = x3 XOR y1
        d = self.genVars_Objective(r)  # 目标变量

        # 添加13个线性约束，模拟模加的差分传播特性
        for i in range(n-1):
            b = [x2[i], x1[i], y0[i]]
            a = [x2[i+1], x1[i+1], y0[i+1]]
            constraints += [a[1]+' - '+a[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' - '+a[1]+' + '+d[i]+' >= 0 ']
            constraints += [a[2]+' - '+a[0]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' + '+d[i]+' <= 3 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' - '+d[i]+' >= 0 ']
            constraints += [b[0]+' + '+b[1]+' + '+b[2]+' + '+d[i]+' - '+a[1]+' >= 0 ']
            constraints += [a[1]+' + '+b[0]+' - '+b[1]+' + '+b[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[1]+' - '+b[0]+' + '+b[1]+' + '+b[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' + '+b[0]+' + '+b[1]+' - '+b[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[2]+' - '+b[0]+' - '+b[1]+' - '+b[2]+' + '+d[i]+' >= -2 ']
            constraints += [b[0]+' - '+a[1]+' - '+b[1]+' - '+b[2]+' + '+d[i]+' >= -2 ']
            constraints += [b[1]+' - '+a[1]+' - '+b[0]+' - '+b[2]+' + '+d[i]+' >= -2 ']
            constraints += [b[2]+' - '+a[1]+' - '+b[0]+' - '+b[1]+' + '+d[i]+' >= -2 ']
        # 处理最后一比特的3路分支情况
        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' <= 2 ']
        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' - 2 temp'+str(r-1)+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x2[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x1[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+y0[n-1]+' >= 0 ']
        
        return constraints

    def genObjectiveFun_to_Round(self, r):
        h = self.BlockSize
        n = int(h/2)
        f = []
        for i in range(1, r+1):
            for j in range(n-1):
                f.append(self.genVars_Objective(i)[j])
        return ' + '.join(f)
    
    def genModel(self, r):
        # 原始方法，从任意输入差分开始搜索
        V = set()
        C = []
        for i in range(1, r+1):
            C += self.genConstraints_of_Round(i)
        V = BasicTools.getVariables_From_Constraints(C)
        add_constraint_1 = ' + '.join(['p'+str(i) for i in range(self.BlockSize)]) + ' >= 1'
        V = V.union(BasicTools.getVariables_From_Constraints([add_constraint_1]))

        filename = os.path.join(output_dir, f'speck{self.BlockSize}diff-round-{r}.lp')
        with open(filename,'w') as o:
            o.write('Minimize\n')
            o.write(self.genObjectiveFun_to_Round(r))
            o.write('\n\nSubject To\n')
            o.write(add_constraint_1+'\n')
            for c in C:
                o.write(c+'\n')
            o.write('\n\nBinary\n')
            for v in V:
                o.write(v+'\n')
        print(f"LP file generated: {filename}")

    def genModel_with_fixed_input(self, r, fixed_bits, diff):
        # 从指定输入差分比特开始搜索
        V = set()
        C = []
        for i in range(1, r+1):
            C += self.genConstraints_of_Round(i)
        V = BasicTools.getVariables_From_Constraints(C)

        input_bits = ['p'+str(i) for i in range(self.BlockSize)]
        for i in fixed_bits:
            C.append(f'{input_bits[i]} = 1')
        for i in range(self.BlockSize):
            if i not in fixed_bits:
                C.append(f'{input_bits[i]} = 0')
        V = V.union(BasicTools.getVariables_From_Constraints(C))

        filename = os.path.join(output_dir, f'speck{self.BlockSize}diff-{diff}-round-{r}-fixed.lp')
        with open(filename,'w') as o:
            o.write('Minimize\n')
            o.write(self.genObjectiveFun_to_Round(r))
            o.write('\n\nSubject To\n')
            for c in C:
                o.write(c+'\n')
            o.write('\n\nBinary\n')
            for v in V:
                o.write(v+'\n')
        print(f"LP file generated: {filename}")
        
        
    def genModel_with_fixed_output(self, r, fixed_bits):
        V = set()
        C = []
        for i in range(1, r+1):
            C += self.genConstraints_of_Round(i)
        V = BasicTools.getVariables_From_Constraints(C)

        output_bits = ['p'+str(i)+'Rd'+str(r) for i in range(self.BlockSize)]

        for i in fixed_bits:
            C.append(f'{output_bits[i]} = 1')
        for i in range(self.BlockSize):
            if i not in fixed_bits:
                C.append(f'{output_bits[i]} = 0')

        V = V.union(BasicTools.getVariables_From_Constraints(C))

        filename = os.path.join(output_dir, f'speck{self.BlockSize}diff-round-{r}-fixed-rev.lp')
        with open(filename,'w') as o:
            o.write('Minimize\n')
            o.write(self.genObjectiveFun_to_Round(r))
            o.write('\n\nSubject To\n')
            for c in C:
                o.write(c+'\n')
            o.write('\n\nBinary\n')
            for v in V:
                o.write(v+'\n')

        print(f"LP file generated: {filename}")


def hex_to_fixed_input_bits(hex_value, bit_width=32):
    fixed_input_bits = []
    binary_str = bin(hex_value)[2:]
    binary_str = binary_str.zfill(bit_width)
    for i, bit in enumerate(binary_str):
        if bit == '1':
            fixed_input_bits.append(i)
    return fixed_input_bits

def main():
    print('Initialized...')
    bar = speck(blocksize)
    for i in range(12):
        diff = 1 << (2 * i + 48 + 17)
        fixed_input_bits = hex_to_fixed_input_bits(diff, bit_width=blocksize) 
        bar.genModel_with_fixed_input(nr, fixed_input_bits, 2 * i + 17)
        # bar.genModel_with_fixed_output(nr, fixed_input_bits)
    


if __name__ == '__main__':
    blocksize = 96
    nr = 11
    output_dir = project_path("milp_output", f"speck{blocksize}")
    os.makedirs(output_dir, exist_ok=True)
    main()
