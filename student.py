from client import Client
import torch, os, copy, wandb
from tqdm import tqdm

LOG_FREQUENCY = 5

class Student(Client):

    def __init__(self,
                client_id: str,
                dataset,
                model,
                batch_size,
                device,
                epochs,
                hyperparam,
                number_classes = None,
                autosave = True,
                wb_log = True,
                server = False,
                teacher = None):
        # use 'super' function to access methos and variables from Client class
        super().__init__(client_id, dataset, model, batch_size, device, epochs, hyperparam, number_classes)
        self.teacher = teacher
        self.is_student = True

    def set_teacher(self, model):
        self.teacher = model

    def get_image_mask(self, prob, pseudo_lab, th=None):
        '''
        Every pixel label that is under the threshold is set to 255 (ignored by the 
        calculations)
        '''
        if th is None:
          th = 0.7
        trash = prob < th
        pseudo_lab[trash==True] = 255

        return pseudo_lab 

    def get_batch_mask(self, outputs):
        # outputs.shape = torch.Size([4, 19, 512, 1024])
        # label shape = torch.Size([4, 1024, 2048])

        prob, pred = outputs.max(dim=1) # retriving the predicted labels
        # prob.shape, pred.shape = torch.Size([4, 1024, 2048]), torch.Size([4, 1024, 2048])
        prob = prob.to(dtype = torch.float32)
        for i in range(len(prob)):
        # normalizing the predicted labels
          '''prob_max = torch.sum(prob[i])
          prob[i] /= prob_max'''
            
          min_val = prob[i]-prob[i].min()
          max_val = min_val.max() 
          prob[i] = min_val/max_val
          
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(prob,pred)], dim=0)
        return mask

    def get_pseudo_lab(self, imgs):
        with torch.no_grad():
          outputs, _, _, _, _ = self.teacher(imgs)
        pseudo_lab = self.get_batch_mask(outputs)

        return pseudo_lab

    def load_teacher(self, load_path):
        if os.path.exists(load_path):
          checkpoint = torch.load(load_path) 
          self.teacher.load_state_dict(checkpoint["model_state_dict"])
          self.teacher.eval()
        else:
          print(f"Model {load_path} not found") 
    
    def run_epoch(self, current_epoch, optimizer, scheduler=None):
        for current_step, (images, _) in enumerate(self.data_loader):
          
          images = images.to(self.device)
          # getting pseudo labels
          labels = self.get_pseudo_lab(images)

          images = images.to(self.device)
          labels = labels.to(self.device, dtype = torch.long)

          optimizer.zero_grad() # Zero-ing the gradients

          outputs, _, _, _, _ = self.model(images)
          loss = self.criterion(outputs, labels) # Compute loss based on output and ground truth

          # mIoU values calculation
          mIoU = self.update_acc(outputs, labels, self.number_classes)

          loss.backward()  # backward pass: computes gradients
          optimizer.step() # update weights based on accumulated gradients

          #logging  
          wandb.log({"Loss": loss.item(),
                    "Mean IoU accuracy": mIoU,
                    "epoch": current_epoch})
          
          if current_step % LOG_FREQUENCY == 0:
            print(f'Step {current_step}, Loss {loss.item()}, Mean IoU accuracy {mIoU}.')
          current_step += 1
      
        scheduler.step()

        print('Using "run_epoch" function from Student class')
    
        return loss, mIoU

    def train(self, epochs=None, hyperparam=None, save_path=None):
      epochs = epochs if epochs is not None else self.epochs
      hyperparam = hyperparam if hyperparam is not None else self.hypers

      optimizer = self.optimizer
      scheduler = self.scheduler

      print(f"ID: {self.client_id} - Training...")
      ep_iterable = tqdm(range(epochs), desc="Epoch: ")
      #ep_iterable = range(epochs)

      self.model.train() #sets module in training mode

      for epoch in ep_iterable:
        #print(f"Epoch {epoch+1}\n-------------------------------")
        loss, mIoU_accuracy = self.run_epoch(epoch, optimizer, scheduler)

        print()

        checkpoint = {
          "epoch": epoch,
          "model_state_dict": self.model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict()
          }
        #finish saving
        if self.server is False:
            self.autosave = False
        if self.autosave is True:
          torch.save(checkpoint,save_path)
      num_samples = len(self.dataset)
      update = copy.deepcopy(self.model.state_dict())

      return num_samples, update