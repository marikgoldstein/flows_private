''' 
   @torch.no_grad()                                                                             
    def evaluate(self, rank, device, which):
        model = self.model if which == 'nonema' else self.ema_model
        model.eval()
     
        self.train_sampler.set_epoch(self.epoch)
        self.test_sampler.set_epoch(self.epoch)
     
        self.logger.info(f"Beginning Eval ...")
        eval_bpd = 0.0
        N = 0
        for batch_idx, (batch, y) in enumerate(self.test_loader):
     
            if self.config.eval_lim_batches is not None and batch_idx >= self.config.eval_lim_batches:
                print("skipping many batches during evaluate")
                break
     
            centered_data, y, ldj = self.prepare_batch(batch, y, device)
            bsz = centered_data.shape[0]                           
            nelbo = self.dgm.loss_fn(centered_data, y, self.model, device, loss_type = 'nelbo')
            bpd_this_batch = self.encoder.nelbo_to_bpd(nelbo, ldj)
            eval_bpd += bpd_this_batch.sum()
            N += bsz
     
        eval_bpd = eval_bpd / N # avg bpd per datapoint on 1 device
        torch.cuda.synchronize()
        #self.logger.info(f"eval bpd for rank {rank} is {eval_bpd}")
        dist.all_reduce(eval_bpd, op=dist.ReduceOp.SUM) # sum axross devices
        eval_bpd = eval_bpd.item() / dist.get_world_size() # avg across device
        #self.logger.info(f"aggregated eval bpd is {eval_bpd}")
        self.logger.info(f"(step={self.train_steps:07d}) Eval BPD: {eval_bpd:.4f}")
        self.maybe_log_wandb({f'{which}_eval_bpd' : eval_bpd}, rank)
     
''' 

