from enum import Enum, unique

@unique
class Types(Enum):
  Input = 'Input'
  Output = 'Output'
  Linear = 'Linear'
  RNN = 'RNN'
  Dropout = 'Dropout'

@unique
class Regs(Enum):
  L1 = 'L1'
  L2 = 'L2'
  No = None
