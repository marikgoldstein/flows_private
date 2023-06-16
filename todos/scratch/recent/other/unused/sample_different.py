
def sampling_only(self,):
	print("sampling only from this ckpt", self.config.resume_path)
	print("self beta func is", self.config.which_beta)
	print("self ts is" ,self.config.backprop_ts)
	self.sampling(with_ema_model=False)
	'''                                                                                                                                                       
		self.config.which_beta = 'const_beta'
		self.config.const_beta = 0.1 + i
		self.sde = VP(
			d=self.d, 
			max_beta = self.config.max_beta, 
			const_beta = self.config.const_beta, 
			which_beta = self.config.which_beta, 
			device=self.config.device
		)
		self.sampling(with_ema_model=False, additional_message=str(i))
	'''


