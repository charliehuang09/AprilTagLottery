class Logger:
  def __init__(self, writter, writter_path):
    self.value = 0
    self.length = 0
    self.writter = writter
    self.writter_path = writter_path
    self.idx = 0
    self.max = None
    self.min = None
    self.write = False
    self.write_prefix = ""
    return
    
  def add(self, input, length):
    input /= length
    self.value += input
    self.length += 1
    return
  
  def get(self):
    if self.length == 0:
      return 0
    output = self.value / self.length
    self.value = 0
    self.length = 0

    if self.write == True:
      self.writter.add_scalar(self.write_prefix + '/' + self.writter_path, output, self.idx)
    if self.max == None:
      self.max = output
    if self.min == None:
      self.min = output

    self.max = max(self.max, output)
    self.min = min(self.min, output)
    self.idx += 1
    return output
  
  def getMax(self):
    return self.max
  
  def getMin(self):
    return self.min

  def clear(self):
    self.value = 0
    self.length = 0
    self.idx = 0
    self.max = None
    self.min = None
    return

  def setWrite(self, input):
    self.write = input
    return
  
  def setPrefix(self, prefix):
    self.write_prefix = prefix
