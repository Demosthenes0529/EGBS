from functools import reduce
import math
import random
import os
from MILPSbox import *
from random import *

blocksize = 32
r1 = 4
r2 = 5
output_dir = f'/home/user/speck_output/speck{blocksize}'
os.makedirs(output_dir, exist_ok=True)

# ------------------ SPECK 正向搜索类 ------------------
class speck():
    def __init__(self, blocksize):
        self.BlockSize = blocksize

    def genVars_InVars_at_Round(self, r):
        assert r >= 1
        if r == 1:
            return ['p'+str(j) for j in range(self.BlockSize)]
        else:      
            return ['p'+str(j)+'Rd'+str(r-1) for j in range(self.BlockSize)]

    def rotl(self, X, n, r):
        temp = [None]*n
        for i in range(n-r):
            temp[i] = X[i+r]
        for i in range(n-r,n):
            temp[i] = X[i-n+r]
        return temp

    def rotr(self, X, n, r):
        temp = [None]*n
        for i in range(r):
            temp[i] = X[n-r+i]
        for i in range(r,n):
            temp[i] = X[i-r]
        return temp

    def genVars_Objective(self, r):
        n = self.BlockSize // 2
        return ['pro'+str(i)+'Rd'+str(r-1) for i in range(n-1)]

    def genConstraints_of_Round(self, r):
        # 正向加法
        assert r>=1
        constraints = []
        X = self.genVars_InVars_at_Round(r)
        Y = self.genVars_InVars_at_Round(r+1)
        n = self.BlockSize // 2
        x0 = X[0:n]
        x1 = X[n:2*n]
        y0 = Y[0:n]
        y1 = Y[n:2*n]

        if n == 16:
            x2 = self.rotr(x0, 16, 7)
            x3 = self.rotl(x1, 16, 2)
        else:
            x2 = self.rotr(x0, n, 8)
            x3 = self.rotl(x1, n, 3)

        constraints += ConstraintGenerator.xorConstraints(x3, y1, y0)
        d = self.genVars_Objective(r)

        for i in range(n-1):
            b = [x2[i], x1[i], y0[i]]
            a = [x2[i+1], x1[i+1], y0[i+1]]
            constraints += [a[1]+' - '+a[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' - '+a[1]+' + '+d[i]+' >= 0 ']
            constraints += [a[2]+' - '+a[0]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' + '+d[i]+' <= 3 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' - '+d[i]+' >= 0 ']

        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' <= 2 ']
        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' - 2 temp'+str(r-1)+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x2[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x1[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+y0[n-1]+' >= 0 ']

        return constraints

    def genObjectiveFun_to_Round(self, r):
        n = self.BlockSize // 2
        f = []
        for i in range(1, r+1):
            for j in range(n-1):
                f.append(self.genVars_Objective(i)[j])
        return ' + '.join(f)

    def genModel(self, r, fixed_bits=None):
        """
        生成LP文件，可指定固定输入比特
        """
        V = set()
        C = []
        for i in range(1, r+1):
            C += self.genConstraints_of_Round(i)
        V = BasicTools.getVariables_From_Constraints(C)

        input_bits = ['p'+str(i) for i in range(self.BlockSize)]
        add_constraint_1 = ' + '.join(input_bits) + ' >= 1'
        C.append(add_constraint_1)

        # 固定输入比特
        if fixed_bits is not None:
            for i in range(self.BlockSize):
                if i in fixed_bits:
                    C.append(f'{input_bits[i]} = 1')
                else:
                    C.append(f'{input_bits[i]} = 0')

        V = V.union(BasicTools.getVariables_From_Constraints(C))
        suffix = 'fixed' if fixed_bits else 'normal'
        filename = os.path.join(output_dir, f'speck{self.BlockSize}diff-round-{r}.lp')
        with open(filename,'w') as o:
            o.write('Minimize\n')
            o.write(self.genObjectiveFun_to_Round(r))
            o.write('\n\nSubject To\n')
            for c in C:
                o.write(c+'\n')
            o.write('\n\nBinary\n')
            for v in V:
                o.write(v+'\n')
        print(f"Forward LP file generated: {filename}")

# ------------------ SPECK 反向搜索类 ------------------
class speck_reverse(speck):
    def genConstraints_of_Round(self, r):
        """
        生成反向搜索的约束
        diff_l + ror(diff_l ^ diff_r, BETA()) = child
        """
        assert r >= 1
        constraints = list()
        X = self.genVars_InVars_at_Round(r)
        Y = self.genVars_InVars_at_Round(r+1)
        n = self.BlockSize // 2
        x0 = X[0:n]
        x1 = X[n:2*n]
        y0 = Y[0:n]
        y1 = Y[n:2*n]

        if n == 16:
            x2 = self.rotl(x0, 16, 7)
            x3 = self.rotr(x1, 16, 2)
        else:
            x2 = self.rotl(x0, n, 8)
            x3 = self.rotr(x1, n, 3)

        constraints += ConstraintGenerator.xorConstraints(x3, y1, y0)
        d = self.genVars_Objective(r)

        for i in range(n-1):
            b = [x2[i], x1[i], y0[i]]
            a = [x2[i+1], x1[i+1], y0[i+1]]
            # 反向加法约束
            constraints += [a[1]+' - '+a[2]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' - '+a[1]+' + '+d[i]+' >= 0 ']
            constraints += [a[2]+' - '+a[0]+' + '+d[i]+' >= 0 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' + '+d[i]+' <= 3 ']
            constraints += [a[0]+' + '+a[1]+' + '+a[2]+' - '+d[i]+' >= 0 ']

        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' <= 2 ']
        constraints += [x2[n-1]+' + '+x1[n-1]+' + '+y0[n-1]+' - 2 temp'+str(r-1)+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x2[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+x1[n-1]+' >= 0 ']
        constraints += ['temp'+str(r-1)+' - '+y0[n-1]+' >= 0 ']

        return constraints

    def genModel_reverse(self, r, fixed_bits=None):
        V = set()
        C = []
        for i in range(1, r+1):
            C += self.genConstraints_of_Round(i)
        V = BasicTools.getVariables_From_Constraints(C)

        input_bits = ['p'+str(i) for i in range(self.BlockSize)]
        add_constraint_1 = ' + '.join(input_bits) + ' >= 1'
        C.append(add_constraint_1)

        if fixed_bits is not None:
            for i in range(self.BlockSize):
                if i in fixed_bits:
                    C.append(f'{input_bits[i]} = 1')
                else:
                    C.append(f'{input_bits[i]} = 0')

        V = V.union(BasicTools.getVariables_From_Constraints(C))
        suffix = 'fixed' if fixed_bits else 'normal'
        filename = os.path.join(output_dir, f'speck{self.BlockSize}diff-round-{r}-rev.lp')
        with open(filename,'w') as o:
            o.write('Minimize\n')
            o.write(self.genObjectiveFun_to_Round(r))
            o.write('\n\nSubject To\n')
            for c in C:
                o.write(c+'\n')
            o.write('\n\nBinary\n')
            for v in V:
                o.write(v+'\n')
        print(f"Reverse LP file generated: {filename}")

# ------------------ 主函数 ------------------
def main():
    print('Initialized...')
    
    # 正向搜索
    forward_speck = speck(blocksize)
    forward_speck.genModel(r1)  # 正向普通搜索
    forward_speck.genModel(r1, fixed_bits=[9])  # 正向固定输入搜索

    # 反向搜索
    reverse_speck = speck_reverse(blocksize)
    reverse_speck.genModel_reverse(r2)  # 反向普通搜索
    reverse_speck.genModel_reverse(r2, fixed_bits=[9])  # 反向固定输入搜索

if __name__ == '__main__':
    main()