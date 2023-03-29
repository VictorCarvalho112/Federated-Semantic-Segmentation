from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch, os, copy, wandb
from mIoU import StreamSegMetrics
from tqdm import tqdm

LOG_FREQUENCY = 50
class Client:
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
              server = False):
      """
      It is not possible to superimpose two classes. Need to figure out how to use 
      run epoch from Student class when is_student == True. 
      super().__init__(self,
                client_id,
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
                teacher = None) """
         
      self.client_id = client_id
      self.dataset = dataset
      self.model = model
      self.batch_size = batch_size
      self.device = device
      self.epochs = epochs
      self.hypers = hyperparam 
      self.number_classes = number_classes if number_classes is not None else 19
      self.autosave = autosave
      self.wb_log = wb_log   
      self.server = server

      self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, drop_last=True)
      

      self.optimizer = optim.SGD(self.model.parameters(), lr=hyperparam["LR"], momentum=hyperparam["MOMENTUM"], weight_decay=hyperparam["WEIGHT_DECAY"])  # optimizing over all the parameters of the BiSeNetV2 net
      self.criterion = nn.CrossEntropyLoss(ignore_index=255) # cross entropy loss

      self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=hyperparam["STEP_SIZE"],gamma=hyperparam["GAMMA"])
      self.best_accuracy = 0  
  ''' 

  def losses(self, images, labels):
    outputs, _, _, _, _ = self.model(images)
    loss = self.criterion(outputs, labels) # Compute loss based on output and ground truth

    return loss, outputs
  '''

  
  def run_epoch(self, current_epoch, optimizer, scheduler=None):
    for current_step, (images, labels) in enumerate(self.data_loader):

      images = images.to(self.device)
      labels = labels.to(self.device, dtype = torch.long)

      optimizer.zero_grad() # Zero-ing the gradients

      outputs, _, _, _, _ = self.model(images)
      loss = self.criterion(outputs, labels) # Compute loss based on output and ground truth
      
      #accuracy
      mIoU_accuracy = self.update_acc(outputs, labels, self.number_classes)

      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients

      #logging  
      wandb.log({"Loss": loss.item(),
                 "Mean IoU accuracy": mIoU_accuracy,
                 "epoch": current_epoch})
      
      if current_step % LOG_FREQUENCY == 0:
        print(f'Step {current_step}, Loss {loss.item()}, Mean IoU accuracy {mIoU_accuracy}.')
      current_step += 1

      checkpoint = {}
      
      if mIoU_accuracy > self.best_accuracy:   #save the best model
        checkpoint = {
          "epoch": current_epoch,
          "model_state_dict": self.model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict()
        }
        self.best_accuracy = mIoU_accuracy
        update = copy.deepcopy(self.model.state_dict())  #create a deep copy of the model state
    
    scheduler.step()

    print('Using "run_epoch" function from Client class')
  
    return loss, self.best_accuracy, checkpoint
  '''    
  def save_model(self, save_path=None, checkpoint=None):
    if save_path is not None and checkpoint is None:
      torch.save(self.model.state_dict(), save_path)  #if a path was passed to this method save it

    elif save_path is None and checkpoint is not None:
      save_dir = self.save_dir                  #If no path was passed use the one from self attributes
      if not os.path.exists(save_dir):          #check if the self.path exists
        os.mkdir(save_dir)            #create a directory if it does not exist
      save_path = os.path.join(save_dir, "epoch" + str(checkpoint["epoch"]) + ".pth")
      torch.save(checkpoint, save_path)
  '''
  def save_model(self, checkpoint=None, save_path=None):
    torch.save(checkpoint, save_path)
    return save_path

  def load_model(self, load_path, state_dict=None):
    if os.path.exists(load_path) and self.server == False:       #check if the load path exists
      checkpoint = torch.load(load_path)    #load the checkpoint from the path
      self.model.load_state_dict(checkpoint["model_state_dict"])  #load the model state
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) #laod optimizer state
      self.model.eval()   #set the model to evaluation
      epoch = checkpoint["epoch"]
      print(f"Loaded model: {load_path}.")
      return epoch
    elif os.path.exists(load_path):
      checkpoint = torch.load(load_path) 
      self.model.load_state_dict(checkpoint["model_state_dict"])
      self.model.eval()
      print(f"Loaded model: {load_path}.")
    else:
      print(f"Model {load_path} not found")  


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
      loss, mIoU_accuracy, checkpoint = self.run_epoch(epoch, optimizer, scheduler)

      print()

      checkpoint = {
        "epoch": epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        }
      #finish saving
      if self.autosave is True:
        torch.save(checkpoint,save_path)
    num_samples = len(self.dataset)
    update = copy.deepcopy(self.model.state_dict())

    return num_samples, update

  def test(self, dataloader=None, cl:str=None):
    self.model.eval() #sets module for inference
    #mIoU_list = []

    #for i ,(images, labels) in enumerate(self.data_loader):
    images, labels = next(iter(self.data_loader))
    images = images.to(self.device, dtype=torch.float32)
    labels = labels.to(self.device, dtype = torch.long)

    outputs, _, _, _, _ = self.model(images)

    _, prediction = outputs.max(dim=1)      
    #mIoU_list.append(self.update_acc(outputs, labels, self.number_classes))
    mIoU = self.update_acc(outputs, labels, self.number_classes)
    self.plot_sample(prediction[0], images[0], labels[0], cl)

    
    #mIoU = sum(mIoU_list)/len(mIoU_list)

    #return mIoU_accuracy, x, prediction, outputs
    return mIoU

  def update_acc(self, outputs, labels, number_classes):
    #mIoU accuracy calculation - from FedDrive repository (function update_metrics)
    x, prediction = outputs.max(dim=1) # retrieving the predicted label
    labels = labels.cpu().numpy()
    prediction = prediction.cpu().numpy()
    miou = StreamSegMetrics(n_classes=number_classes)
    miou.update(labels,prediction)
    mIoU_accuracy = miou.get_results()["Mean IoU"]
    miou.reset()

    return mIoU_accuracy

  def plot_sample(self, prediction, image, label, cl: str=None):

    image = image.cpu()
    label = label.cpu()
    prediction = prediction.cpu()

    found = True
    if cl is not None:
      map_classes = self.dataset.map_classes
      if cl in map_classes.values():
        for cl, name in enumerate(map_classes.values()):
          if name == cl:
            mapping_pred = prediction==cl
            mapping_label = label==cl
      else:
        print("Class not found")
        flag = True
    
    elif cl is None or not found: 
      mapping_pred = prediction!=255
      mapping_label = label!=255

    plt.imshow(image.permute(1,2,0))
    plt.show()
            
    plt.imshow(prediction*mapping_pred+1, cmap="gray")
    plt.show()
            
    plt.imshow(label*mapping_label+1, cmap="gray")
    plt.show()