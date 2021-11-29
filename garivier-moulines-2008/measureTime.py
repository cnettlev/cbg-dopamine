import time
import inspect

class measure:

	def __init__(self):
		self.caller = inspect.stack()[1][3]
		self.init = time.time()

	def stop(self):
		print "Function ",self.caller,"took",time.time()-self.init