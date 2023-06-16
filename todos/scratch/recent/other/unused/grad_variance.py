
def _measure_grad_variance(self, batch, is_train):
	assert False 
	with_ema_model = False
	var_dict = {}
	N = batch.shape[0]
	for loss_type in ['nelbo', 'nelbo_st']:

		var_dict[loss_type] = {}
		for n, p in self.model_obj.model.named_parameters():
			var_dict[loss_type][n] = []
		

		u_0, ldj = self.encoder.preprocess(batch)
		for i in range(N):
			ui = u_0[i]
			ui = ui[None, ...]
			assert ui.shape == (1, 28*28)
			t, w = self.time_min_one.sample(1,)
			assert t.shape == (1,), t.shape
			D = self.sde(ui, t) 
			assert torch.allclose(D['t'], t) # just a debug
			# prior term
			prior = self.sde.cross_ent_helper(ui)
			if loss_type == 'nelbo':
				int_t = w * self.dsm(D, with_ema_model)
				nelbo = -(prior + int_t)
				assert prior.shape == (1,)
				assert int_t.shape == (1,)
				loss = nelbo
			else:
				int_st = self.loss_st(ui, 'dsm', with_ema_model, is_train)
				nelbo_st = -(prior + int_st)
				assert prior.shape == (1,)
				assert int_st.shape == (1,)
				loss = nelbo_st
			
			assert loss.shape == (1,)
			self.model_obj.compute_grads_no_step(loss)
			for n, p in self.model_obj.model.named_parameters():
				var_dict[loss_type][n].append(p.grad)
	return var_dict

