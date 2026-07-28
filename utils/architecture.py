def bitfieldOut(n, extend=8):
    return [int(digit) for digit in bin(n)[2:].zfill(extend)] # [2:] to chop off the "0b" part

def bitfieldIn(n):
    arr = []
    idx = int(n)
    if (idx > 7 or idx < 0):
        assert(False)
    arr = [0]*8
    arr[idx] = 1
    return arr

class Architecture:
    def __init__(self, rf_in, rf_out, rf_size, lsu_in1, lsu_in2, lsu_out, alu_out, alu2_enable,
                 alu2_in1, alu2_in2, alu2_out,
                 muldiv_type, muldiv_in1, muldiv_in2, muldiv_out):
        self.rf_in          = rf_in
        self.rf_out         = rf_out
        self.rf_size        = rf_size
        self.lsu_in1        = lsu_in1
        self.lsu_in2        = lsu_in2
        self.lsu_out        = lsu_out
        self.alu_out        = alu_out
        self.alu2_enable    = alu2_enable
        self.alu2_in1       = alu2_in1
        self.alu2_in2       = alu2_in2
        self.alu2_out       = alu2_out
        self.muldiv_type    = muldiv_type
        self.muldiv_in1     = muldiv_in1
        self.muldiv_in2     = muldiv_in2
        self.muldiv_out     = muldiv_out

        if alu2_enable == 0:
            self.alu2_in1 = 0
            self.alu2_in2 = 0
            self.alu2_out = 0

    # Getter methods for each member variable
    def print_arch(self):
        print("rf_in=%1d rf_out=%1d rf_size=%3d lsu_in1=%1d \
lsu_in2=%1d lsu_out=%3d alu_out=%3d alu2_enable=%1d \
alu2_in1=%1d alu2_in2=%1d alu2_out=%3d muldiv_type=%1d \
muldiv_in1=%1d muldiv_in2=%1d muldiv_out=%3d" %
              (self.rf_in, self.rf_out, self.rf_size, self.lsu_in1,
               self.lsu_in2, self.lsu_out, self.alu_out, self.alu2_enable,
               self.alu2_in1, self.alu2_in2, self.alu2_out,
               self.muldiv_type, self.muldiv_in1, self.muldiv_in2,
               self.muldiv_out))
    def get_rf_size(self):
        return self.rf_size

    def get_rf_in(self):
        return self.rf_in

    def get_rf_out(self):
        return self.rf_out

    def get_lsu_in1(self):
        return self.lsu_in1

    def get_lsu_in2(self):
        return self.lsu_in2

    def get_lsu_out(self):
        return self.lsu_out

    def get_alu_out(self):
        return self.alu_out

    def get_muldiv_type(self):
        return self.muldiv_type

    def get_muldiv_in1(self):
        return self.muldiv_in1

    def get_muldiv_in2(self):
        return self.muldiv_in2

    def get_muldiv_out(self):
        return self.muldiv_out

    def get_alu2_enable(self):
        return self.alu2_enable

    def get_alu2_in1(self):
        return self.alu2_in1

    def get_alu2_in2(self):
        return self.alu2_in2
    
    def get_alu2_out(self):
        return self.alu2_out

    def getParams(self):
        vars = []
        vars.append(self.rf_in)
        vars.append(self.rf_out)
        vars.append(self.rf_size)
        vars.append(self.lsu_in1)
        vars.append(self.lsu_in2)
        vars.append(self.lsu_out)
        vars.append(self.alu_out)
        vars.append(self.alu2_enable)
        if self.alu2_enable:
            vars.append(self.alu2_in1)
            vars.append(self.alu2_in2)
            vars.append(self.alu2_out)
        else:
            vars.extend([0]*3)
        vars.append(self.muldiv_type)
        vars.append(self.muldiv_in1)
        vars.append(self.muldiv_in2)
        vars.append(self.muldiv_out)
        assert(len(vars) == 15)
        return vars

    
    def getModelVector(self):
        vars = []
        vars.append(self.rf_in)
        vars.append(self.rf_out)
        vars.append(self.rf_size)
        vars.extend(bitfieldIn(self.lsu_in1))
        vars.extend(bitfieldIn(self.lsu_in2))
        vars.extend(bitfieldOut(self.lsu_out))
        vars.extend(bitfieldOut(self.alu_out))
        vars.append(self.alu2_enable)
        assert(self.alu2_enable == 1 or self.alu2_enable == 0)
        if self.alu2_enable == 1:
            vars.extend(bitfieldIn(self.alu2_in1))
            vars.extend(bitfieldIn(self.alu2_in2))
            vars.extend(bitfieldOut(self.alu2_out))
        else:
            vars.extend([0]*24)
        assert(self.muldiv_type == 1 or self.muldiv_type == 2)
        if self.muldiv_type == 1:
            vars.append(1)
            vars.append(0)
        else:
            vars.append(0)
            vars.append(1)
        vars.extend(bitfieldIn(self.muldiv_in1))
        vars.extend(bitfieldIn(self.muldiv_in2))
        vars.extend(bitfieldOut(self.muldiv_out))
        assert(len(vars) == 86)
        return vars

    def writeToFile(self, path):
        with open(path, 'w') as f:
            f.write("RF_IN_AMOUNT," + str(self.get_rf_in()) + "\n")
            f.write("RF_OUT_AMOUNT," + str(self.get_rf_out()) + "\n")
            f.write("RF_SIZE," + str(self.get_rf_size()) + "\n")
            f.write("LSU_IN1," + str(self.get_lsu_in1()) + "\n")
            f.write("LSU_IN2," + str(self.get_lsu_in2()) + "\n")
            f.write("LSU_BP_OUT," + str(self.get_lsu_out()) + "\n")
            f.write("ALU_BP_OUT," + str(self.get_alu_out()) + "\n")
            f.write("ALU2_ENABLE," + str(self.get_alu2_enable()) + "\n")
            f.write("ALU2_IN1," + str(self.get_alu2_in1()) + "\n")
            f.write("ALU2_IN2," + str(self.get_alu2_in2()) + "\n")
            f.write("ALU2_BP_OUT," + str(self.get_alu2_out()) + "\n")
            f.write("MULDIV_TYPE," + str(self.get_muldiv_type()) + "\n")
            f.write("MULDIV_IN1," + str(self.get_muldiv_in1()) + "\n")
            f.write("MULDIV_IN2," + str(self.get_muldiv_in2()) + "\n")
            f.write("MULDIV_BP_OUT," + str(self.get_muldiv_out()) + "\n")

