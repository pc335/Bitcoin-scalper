class DQN(nn.Module):
    
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=6):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def select_action(state, steps_done,eps_decay=1000):
    eps_threshold = 0.05 + 0.96 * np.exp(-steps_done / eps_decay )
    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bsim = Bitcoin_simulator(100000,64)
num_episodes = 500
chunklist = chunkate(updated_df,64)
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
target_update = 10
actions = ["BUY","SELL","STAY","QUIT","FURTHER BUY","FURTHER SELL"]
n_actions=len(actions)
dict_actions = {key: value for key,value in enumerate(actions)}
filter_action = {0:torch.tensor([1,1,0,0,0,0],dtype=torch.float32,device=device),1:torch.tensor([1,1,0,0,0,0],dtype=torch.float32,device=device),2:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),3:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),4:torch.tensor([0,0,0,0,0,0],dtype=torch.float32,device=device),5:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device),6:torch.tensor([0,0,1,1,1,1],dtype=torch.float32,device=device)}

for batch in chunklist:
    bsim.state_array(batch)

    for epis in range(num_episodes):
        first_action = random.choice([0, 1])
        first_state = torch.tensor(bsim.array_s[0], dtype=torch.float32, device=device).view(1,-1)
        print(f" state size {first_state.size()}")

        reward, state, quit = bsim.get_state(first_action, first_state)
        print(f"state is {state}")
       
        state = torch.tensor(state, dtype=torch.float32, device=device).view(1,-1)
        print(f"state is {state.size()}")
        q_values = policy_net(first_state).gather(1, torch.tensor([first_action], device=device).view(-1,1))
        
        masked_qv = q_values * filter_action[first_action][first_action]

        with torch.no_grad():
            max_next_q = target_net(state)
            max_next_filt = max_next_q * filter_action[first_action]
            target_q_values = reward + (0.99 * max_next_filt.max().item() * (1 - quit))
        
        loss = F.mse_loss(masked_qv, torch.tensor([target_q_values], dtype=torch.float32, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        first_state = state.clone()
        first_action = select_action(first_state, 0)
        print(f"first action{first_action}")

        for i in range(300):
            
            reward, state, quit = bsim.get_state(first_action.item(), first_state)
            state = torch.tensor(state, dtype=torch.float32, device=device).view(1,-1)
            q_values = policy_net(first_state)
            print(f"qvalues{q_values.size()}")
            q_values = policy_net(first_state).gather(1, first_action.view(1,-1))
            masked_qv = q_values * filter_action[first_action.item()][first_action.item()]

            with torch.no_grad():
                max_next_q = target_net(state)
                max_next_filt = max_next_q * filter_action[first_action.item()]
                target_q_values = reward + (0.99 * max_next_filt.max().item() * (1 - quit))

            loss = F.mse_loss(masked_qv, torch.tensor([target_q_values], dtype=torch.float32, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if quit:
                break

            first_state = state.clone()
            first_action = select_action(first_state, i)
