class ConstraintGenerator:

    @staticmethod
    def xorConstraints(x, y, z):
        """
        x XOR y = z
        """
        constraints = []
        for i in range(len(x)):
            constraints.append(f"{x[i]} + {y[i]} - {z[i]} >= 0")
            constraints.append(f"{x[i]} - {y[i]} + {z[i]} >= 0")
            constraints.append(f"- {x[i]} + {y[i]} + {z[i]} >= 0")
            constraints.append(f"{x[i]} + {y[i]} + {z[i]} <= 2")
        return constraints


class BasicTools:

    @staticmethod
    def getVariables_From_Constraints(C):
        V = set()
        for s in C:
            temp = s.strip()
            temp = temp.replace('+', ' ')
            temp = temp.replace('-', ' ')
            temp = temp.replace('>=', ' ')
            temp = temp.replace('<=', ' ')
            temp = temp.split()

            for v in temp:
                if not v.isdigit():
                    V.add(v)
        return V





def three_way_fork(x, y, z, r):
    """
    Constranints for three-fork branch
    X + Y + Z - 2 dummy >= 0
    X + Y + Z <= 2
    dummy - X >= 0
    dummy - Y >= 0
    dummy - Z >= 0
    """
    constraints = []
    for i in range(len(x)):
        constraints += [x[i] + ' + ' + y[i] + ' + ' + z[i] + ' <= 2']
        constraints += [x[i] + ' + ' + y[i] + ' + ' + z[i] + ' - 2 temp' + str(i) + 'Rnd' + str(r-1) + ' >= 0']
        constraints += ['temp' + str(i) + "Rnd" + str(r-1) + ' - ' + x[i] + ' >= 0']
        constraints += ['temp' + str(i) + "Rnd" + str(r-1) + ' - ' + y[i] + ' >= 0']
        constraints += ['temp' + str(i) + "Rnd" + str(r-1) + ' - ' + z[i] + ' >= 0']
    return constraints


def getVariables_From_Constraints(C):
    V = set([])
    for s in C:
        temp = s.strip()
        temp = temp.replace('+', ' ')
        temp = temp.replace('-', ' ')
        temp = temp.replace('>=', ' ')
        temp = temp.replace('<=', ' ')
        temp = temp.split()
    for v in temp:
        if not v.isdigit():
            V.add(v)
    return V














