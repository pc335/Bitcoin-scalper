def select_action(state, steps_done,eps_decay=1000):
    eps_threshold = 0.05 + 0.96 * np.exp(-steps_done / eps_decay )
    if random.random() < eps_threshold:
        return random.randrange(n_actions)
        #return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bsim = Bitcoin_simulator(100000,64)
num_episodes=500
chunklist = chunkate(updated_df,64)
chunkepis_list=[[len(chunk),chunk] for chunk in chunklist]
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
target_update = 10
loss_update=100
actions = ["BUY","SELL","STAY","QUIT","FURTHER BUY","FURTHER SELL"]
n_actions=len(actions)
dict_actions = {key: value for key,value in enumerate(actions)}
filter_action = {0:torch.tensor([1,1,0,0,0,0],dtype=torch.float32,device=device),1:torch.tensor([1,1,0,0,0,0],dtype=torch.float32,device=device),2:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),3:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),4:torch.tensor([0,0,0,0,0,0],dtype=torch.float32,device=device),5:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),6:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device)}
update_epis=100
for rangec in range(len(chunklist)):
    
    chunkl=chunklist.copy()
    batch=random.sample(chunkepis_list,1)[0]

    bsim.state_array(batch[1])
    for epis in range(num_episodes):
        if epis%update_epis==0:
            print(f"epsi is {epis}")
            print(f" trade log {bsim.trade_log}")
        bsim.reset()
         
        first_action = random.choice([0, 1])
        first_state = bsim.array_s[0]
        

        reward, state, quit = bsim.get_state(first_action, first_state)
      
       
        #state = torch.tensor(state, dtype=torch.float32, device=device).view(1,-1)
      
        q_values = policy_net(torch.tensor(state, dtype=torch.float32, device=device).view(1,-1)).gather(1, torch.tensor([first_action], device=device).view(-1,1))
        
        masked_qv = q_values * filter_action[first_action][first_action]

        with torch.no_grad():
            max_next_q = target_net(torch.tensor(state, dtype=torch.float32, device=device).view(1,-1))
            max_next_filt = max_next_q * filter_action[first_action]
            target_q_values = reward + (0.99 * max_next_filt.max().item() * (1 - quit))
        
        loss = F.mse_loss(masked_qv, torch.tensor([target_q_values], dtype=torch.float32, device=device).view(1,-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        first_state = state
        first_action = select_action(first_state, 0)
      
        bsim.iters=0
       
        for i in range(batch[0]):
            
            if i==batch[0]-1:
            
                bsim.iters-=1
                print(bsim.iters)
                
            reward, state, quit = bsim.get_state(first_action, first_state)
            
            
           
            q_values = policy_net(torch.tensor(state, dtype=torch.float32, device=device).view(1,-1)).gather(1, torch.tensor(first_action,dtype=torch.int64, device=device).view(1,-1))
            masked_qv = q_values * filter_action[first_action][first_action]

            with torch.no_grad():
                max_next_q = target_net(torch.tensor(state, dtype=torch.float32, device=device).view(1,-1))
                max_next_filt = max_next_q * filter_action[first_action]
                target_q_values = reward + (0.99 * max_next_filt.max().item() * (1 - quit))

            loss = F.mse_loss(masked_qv, torch.tensor([target_q_values], dtype=torch.float32, device=device).view(1,-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if i%loss_update==0:
                print(f"loss at iters{i} {loss}")
            if quit==1:
                break

            first_state = state
            first_action = select_action(first_state, i)
policy_net_save = "C:/Users/Lenovo/dqn_bitcoin_weights3.pth"
torch.save(policy_net.state_dict(), policy_net_save)

       
