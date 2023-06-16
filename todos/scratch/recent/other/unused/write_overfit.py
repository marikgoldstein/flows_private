def write_overfit_batch(self,):                                                                                                                               
	print("writing overfit batch")
	batch_idx, (batch, _ ) = next(enumerate(self.train_loader))
	'''
	xi = batch[0]
	xj = batch[1]
	assert xi.shape == (3, 32, 32)
	assert xj.shape == (3, 32, 32)
	xi = xi.unsqueeze(0).repeat(64, 1, 1, 1)
	xj = xj.unsqueeze(0).repeat(64, 1, 1, 1)
	assert xi.shape == (64, 3, 32, 32)
	assert xj.shape == (64, 3, 32, 32)
	x = torch.cat([xi,xj],dim=0)
	assert x.shape == (128, 3, 32, 32)
	'''
	x = batch
	torch.save(x, 'overfit_batch.pt')
	print("wrote overfit batch")
	assert False

