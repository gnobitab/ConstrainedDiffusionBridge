import torch
import numpy as np
import torch.nn.functional as F
import adult_loader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import math

def index_to_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order).float()

    return x_onehot

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=2, hidden_num=100):
        super().__init__()
        self.t_embedding = nn.Linear(1, hidden_num, bias=True)
        self.fc_in = nn.Linear(input_dim+hidden_num, hidden_num, bias=True)
        self.bn_in = nn.BatchNorm1d(hidden_num)
        self.fc_list = [nn.Sequential(nn.Linear(hidden_num + hidden_num, hidden_num, bias=True), 
                                      nn.BatchNorm1d(hidden_num))
                        for i in range(hidden_layers)]
        self.fc_list = nn.ModuleList(self.fc_list)
        self.fc_out = nn.Linear(hidden_num + hidden_num, input_dim, bias=True)
        self.act = lambda x: F.leaky_relu(x)
    
    def forward(self, x_input, t):
        t = self.t_embedding(t)
        t = self.act(t)
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc_in(inputs)
        x = self.bn_in(x)
        x = self.act(x)
        x = torch.cat([x, t], dim=1)
        for fc in self.fc_list:
            x = fc(x)
            x = self.act(x)
            x = torch.cat([x, t], dim=1)
        x = self.fc_out(x)

        return x

class ConstrainedBrownianBridge():
  def __init__(self, data_shape=None, 
                     x0=0.0, 
                     sigma_min=0.01, 
                     sigma_max=50, 
                     N=1000, 
                     init_type='const', 
                     noise_type='ddpm', 
                     with_noise_decay=True):
    """Construct a Constrained Brownian Bridge.
    Args:
      N: number of discretization steps
    """

    self.data_shape = data_shape
    self.N = N
    self.x0 = x0
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.with_noise_decay = with_noise_decay

    self.model = MLP(input_dim=self.data_shape[1], hidden_layers=1, hidden_num=512)

    self.init_type = init_type
    self.noise_type = noise_type
    self.discrete_sigmas, self.sampling_process_type = self.get_noise(noise_type) 
 
  def get_x0(self, init_type, n=None):
      assert n is not None

      if init_type == 'const': 
          init = torch.zeros(self.data_shape) 
          return init.repeat(n, 1)

      elif init_type == 'gaussian':
          ### standard gaussian 
          cur_shape = (n, self.data_shape[1])
          return torch.randn(cur_shape) 
      
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED")

  def get_noise(self, noise_type):
      if noise_type == 'ddpm':  
          ### type1: ddpm style,  - b * exp(-ax), direct formula for update
          sigmas = torch.exp(torch.linspace(np.log(self.sigma_max), np.log(self.sigma_min), self.N))
          sigmas_square_sum = [sigmas[0:1]**2]
          for i in range(self.N-1):
              sigmas_square_sum.append(sigmas[(i+1):(i+2)]**2 + sigmas_square_sum[-1])
          sigmas_square_sum = torch.cat(sigmas_square_sum) 

          return [sigmas, sigmas_square_sum], 'direct'

      elif noise_type == 'exp':
          ### type2: - b * exp(-ax), variance reduction process for update 
          ### direvative: ab exp(-ax), beta_T = -b exp(-a)
          ### we use sigma_max for a, sigma_min for b
          time = torch.linspace(0, 1.-1./self.N, self.N)
          sigmas = self.sigma_min * self.sigma_max * torch.exp((-1.)*self.sigma_max*time)
          beta = (-1.)*self.sigma_min*torch.exp((-1.)*self.sigma_max*time)
          bias = beta[0]
          beta = beta - bias
          beta_T = (-1.)*self.sigma_min*np.exp((-1.)*self.sigma_max) - bias
          denominators = beta_T - beta 

          return [sigmas, denominators, beta_T, beta], 'change_variance' 
      
      elif noise_type == 'linear':
          ### type3:b(2-t)t/2, variance reduction process for update 
          ### direvative: b(1-t)+c, beta_T = b/2 + c
          ### we use sigma_min for b
          c = 0.1
          time = torch.linspace(0, 1.-1./self.N, self.N)
          sigmas = self.sigma_min * (1. - time) + c
          beta = self.sigma_min * (2. -time) * time /2. + c * time
          bias = beta[0]
          beta = beta -bias
          beta_T = self.sigma_min/2. - bias + c
          denominators = beta_T - beta 

          return [sigmas, denominators, beta_T, beta], 'change_variance' 
     
      elif noise_type == 'inverse_exp':
          ### type4: abt-b exp(at-a)+ct, variance reduction process for update 
          ### direvative: ab-ab exp(-a(1-x))+c, beta_T = ab - b +c
          ### we use sigma_max for a, sigma_min for b
          c = 0.1
          time = torch.linspace(0, 1.-1./self.N, self.N)
          sigmas = self.sigma_min * self.sigma_max * (1. - torch.exp((-1.)*self.sigma_max*(1.-time))) + c
          beta = self.sigma_min * self.sigma_max * time + (-1.)*self.sigma_min*torch.exp((-1.)*self.sigma_max*(1.-time)) + c * time
          bias = beta[0]
          beta = beta -bias
          beta_T = self.sigma_min * self.sigma_max - self.sigma_min - bias + c
          denominators = beta_T - beta 

          return [sigmas, denominators, beta_T, beta], 'change_variance' 
 
      else:
          raise NotImplementedError("NOISE TYPE NOT IMPLEMENTED")

  def get_denominator(self, epsilon, t):
      if self.with_noise_decay:
          if self.sampling_process_type == 'direct':
              denominator = 1.-epsilon*t
          elif self.sampling_process_type == 'change_variance':
              denominator = self.discrete_sigmas[1][t]
      else:
          denominator = 1.-epsilon*t
      
      return denominator   
  
  @property
  def T(self):
    return 1.
   

  @torch.no_grad()
  def get_perturbed_data(self, batch):
    '''
        Sample from the intermediate conditional distributions
    '''
    n, c = batch.shape
    
    x0 = self.get_x0(self.init_type, n).detach().clone().to(batch.device) 
    labels = torch.randint(0, self.N, (batch.shape[0],), device=batch.device) ### NOTE: label for time
    time = labels.view(-1, 1).to(batch.device)

    ## NOTE: get conditional distribution with random times 
    noise = torch.randn_like(x0, device=x0.device)
    
    if self.with_noise_decay:
        if self.sampling_process_type == 'change_variance':

            sigmas, beta_T, beta = self.discrete_sigmas[0], self.discrete_sigmas[2], self.discrete_sigmas[3]
            epsilon = 1./self.N
            beta = beta[labels].to(batch.device)
            beta_T_minus_beta = beta_T - beta
            beta = beta.view(-1, 1).repeat(1,c)
            perturbed_data = (beta * batch + (beta_T - beta) * x0)/beta_T
            denominator = self.get_denominator(epsilon, labels).to(batch.device) / self.discrete_sigmas[0][labels].to(batch.device)
            perturbed_data = perturbed_data  + noise * torch.sqrt(beta * (1. - beta/beta_T))

        else:
            raise NotImplementedError("SAMPLING PROCESS TYPE NOT IMPLEMENTED")
    else:

        perturbed_data = time * (1./self.N) * batch + (1. - time * (1./self.N)) * x0
        denominator = self.get_denominator(1./self.N, labels)
        perturbed_data = perturbed_data  + noise * torch.sqrt((time * (1./self.N)) * (1.-time * (1./self.N)))
    
    b = (batch - perturbed_data).detach().clone() / denominator.view(-1, 1).repeat(1, c) ### b = sigma_t **2 * (x-z)/ (beta_T - beta_t) after denominator
            
    with torch.enable_grad():
        grad_log_h = torch.zeros_like(perturbed_data, device=perturbed_data.device)
        discrete_pointer = self.data_info['discrete_pointer']
        continuous_pointer = self.data_info['continuous_pointer']

        for col in self.data_info['order']:
            ### NOTE: cur = current
            if self.data_info[col]['type'] == 'discrete':

                indices = torch.tensor([i for i in range(self.data_info[col]['length'])])
                indices = F.one_hot(indices, self.data_info[col]['length'])
                        
                inputs = perturbed_data.detach().requires_grad_(True)
                t_cur = 1./self.N * labels

                log_h_omega = self.get_log_h_omega_discrete(inputs, t_cur, indices, beta_T_minus_beta,
                                                            self.data_info[col], 
                                                            start_ind=discrete_pointer, 
                                                            end_ind=discrete_pointer+self.data_info[col]['length'])

                grad_log_h = grad_log_h + torch.autograd.grad(log_h_omega.sum(), inputs, allow_unused=True, retain_graph=False)[0]
                discrete_pointer += self.data_info[col]['length']

            elif self.data_info[col]['type'] == 'continuous':
                if self.data_info[col]['bound']:
                    inputs = perturbed_data.detach().requires_grad_(True)
                    t_cur = 1./self.N * labels

                    log_h_omega = self.get_log_h_omega_continuous_non_negative(inputs, t_cur, beta_T_minus_beta, ind=continuous_pointer)

                    grad_log_h = grad_log_h + torch.autograd.grad(log_h_omega.sum(), inputs, allow_unused=True, retain_graph=False)[0]
                continuous_pointer += 1    

            else:
                assert False, 'Only support discrete or continuous'
    
    b = b - sigmas[labels].to(batch.device)[:, None] * grad_log_h
    
    labels = labels.view(-1, 1)/self.N 

    return perturbed_data, b, labels 
  
  def get_log_h_omega_discrete(self, z, t, indices, beta_T_minus_beta, data_info, start_ind=0, end_ind=0):
    num_classes = data_info['length']
    z_sub = z[:, start_ind:end_ind] # subset
    indices = indices.to(z.device).view(1, num_classes, num_classes).repeat(z.shape[0], 1, 1)
    z_prime = z_sub.view(z_sub.shape[0], 1, z_sub.shape[1]).repeat(1, num_classes, 1)
    
    t = t.view(-1, 1).repeat(1, z_sub.shape[1])
    
    
    log_h = (-1.) * (z_prime - indices).norm(dim=2).pow(2) /(2.*beta_T_minus_beta[:, None])
    log_h_max, idx = torch.max(log_h, dim=1)
    log_h_max = log_h_max.view(z.shape[0], 1).repeat(1, num_classes)
    log_h = log_h - log_h_max.detach()
    log_h = log_h.exp()
    log_h = log_h.sum(dim=1, keepdim=False)

    return log_h.log()

  def get_log_h_omega_continuous_non_negative(self, z, t, beta_T_minus_beta, ind=0):
    def std_gaussian_cdf(value):
        return 0.5 * (1 + torch.erf(value / math.sqrt(2.)))

    z_sub = z[:, ind:(ind+1)]
    
    ### to be non-negative, we need the value to be larger than -1. we found that slightly large than -1 helps ensuring the constraint, so we set 0.95
    cdf = std_gaussian_cdf((z_sub + 0.95)/torch.sqrt(beta_T_minus_beta[:, None]))

    return torch.log(cdf+1e-9)

  @torch.no_grad()
  def sample(self, n, device='cuda'):
      ### NOTE: n is the number of samples
      self.model.eval()
      x0 = self.get_x0(self.init_type, n).detach().clone().to(device)
      _, c = x0.shape
      epsilon_base = 1./self.N
      z = x0.clone()

      for t in range(self.N):
        noise = torch.randn_like(z, device=z.device)

        if self.with_noise_decay:

            if self.sampling_process_type == 'direct':

                epsilon = epsilon_base
                denominator = self.get_denominator(epsilon, t)
                noise = noise * self.discrete_sigmas[0][t]

            elif self.sampling_process_type == 'change_variance':
               
                sigmas, beta_T, beta = self.discrete_sigmas[0], self.discrete_sigmas[2], self.discrete_sigmas[3]
                beta = beta[t]
                beta_T_minus_beta = (beta_T - beta) * torch.ones((n), device=z.device)
            
                epsilon = epsilon_base  
                noise = torch.sqrt(sigmas[t]) * noise 
                denominator = self.get_denominator(epsilon, t) / sigmas[t]

            else:
                raise NotImplementedError("SAMPLING PROCESS TYPE NOT IMPLEMENTED")
        else:
            ### No noise
            epsilon = epsilon_base
            denominator = self.get_denominator(epsilon, t)

        denominator = denominator.to(device)
        t_tensor = torch.ones((n), device=z.device) * t  
        z, t_tensor = z.float(), t_tensor.float()

        with torch.enable_grad():
            grad_log_h = torch.zeros_like(z, device=z.device)
            discrete_pointer = self.data_info['discrete_pointer']
            continuous_pointer = self.data_info['continuous_pointer']

            for col in self.data_info['order']:
                ### NOTE: cur = current
                if self.data_info[col]['type'] == 'discrete':
                    #print(discrete_pointer, self.data_info[col]['length'])
                    indices = torch.tensor([i for i in range(self.data_info[col]['length'])])
                    indices = F.one_hot(indices, self.data_info[col]['length'])
                            
                    inputs = z.detach().clone().requires_grad_(True)
                    t_cur = 1./self.N * t_tensor

                    log_h_omega = self.get_log_h_omega_discrete(inputs, t_cur, indices, beta_T_minus_beta,
                                                                self.data_info[col], 
                                                                start_ind=discrete_pointer, 
                                                                end_ind=discrete_pointer+self.data_info[col]['length'])

                    grad_log_h = grad_log_h + torch.autograd.grad(log_h_omega.sum(), inputs, allow_unused=True, retain_graph=False)[0]
                    discrete_pointer += self.data_info[col]['length']

                elif self.data_info[col]['type'] == 'continuous':

                    if self.data_info[col]['bound']:
                        inputs = z.detach().clone().requires_grad_(True)
                        t_cur = 1./self.N * t_tensor

                        log_h_omega = self.get_log_h_omega_continuous_non_negative(inputs, t_cur, beta_T_minus_beta, ind=continuous_pointer)

                        grad_log_h = grad_log_h + torch.autograd.grad(log_h_omega.sum(), inputs, allow_unused=True, retain_graph=False)[0]
                    continuous_pointer += 1    

                else:
                    assert False, 'Only support discrete or continuous'

        t_tensor = t_tensor.view(-1, 1) / self.N
        pred = self.model(z, t_tensor)
        z = z.detach().clone() + epsilon * (pred + sigmas[t] * grad_log_h)  + np.sqrt(epsilon) * noise
        
      print('Examplery Sample No.0:', z[0])
      
      discrete_pointer = self.data_info['discrete_pointer']
      for col in self.data_info['order']:
          if self.data_info[col]['type'] == 'discrete':
              print('Discrete Column:', col)
              print('Generated:', z[0][discrete_pointer:(discrete_pointer+self.data_info[col]['length'])].cpu().numpy())
              discrete_pointer += self.data_info[col]['length']
      
      out = adult_loader.inverse_transform_numpy_to_pandas(z.cpu().numpy(), self.data_info)
      return out  
  
  def fit(self, data, discrete_columns, continuous_constraint_columns, iterations=1000, batchsize=128, device='cuda'):
      data_np, self.data_info = adult_loader.get_training_data_and_info(data, discrete_columns, continuous_constraint_columns)
      self.model = self.model.to(device)
      optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
      self.model.train()

      for i in range(iterations+1):
          optimizer.zero_grad()
          indices = torch.randperm(len(data_np))[:batchsize]
          batch = torch.tensor(data_np[indices], device=device)
          x, b, t = self.get_perturbed_data(batch)
          x, b, t = x.float(), b.float(), t.float()
          out = self.model(x, t)

          loss = (out - b).view(b.shape[0], -1).pow(2).sum(dim=1)
          loss = loss.mean()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
          optimizer.step()
          if i % 100 == 0:
              print('Iteration:', i, 'Training Loss:', loss.item())
      print('Training Finished!')    
      
