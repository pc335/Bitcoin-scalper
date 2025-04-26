@dataclass
class Trade:
    Price: float
    Volume: float
    Tradeside: str
    Net_Profit: float
    Active: str
    failed_Trade: str

class Bitcoin_simulator:
    def __init__(self, funds, chunk):
        self.funds = funds / 2
        self.ltp = 0
        self.vol = 0
        self.further_trades = {}
        self.iters = 0
        self.trade_log = []
        self.trade_index = None

    def reset(self):
        self.ltp = 0
        self.vol = 0
        self.further_trades = {}
        self.iters = 0
        self.trade_log = []
        self.trade_index = None

    def get_state(self, action, next_state, nest=False):
        self.iters += 1
        print(f"action inside is {action}")
        price = self.array_s[self.iters, 3] if action== 1 else self.array_s[self.iters, 5]
        volume = self.array_s[self.iters, 4] if action == 1 else self.array_s[self.iters, 6]
        Tradeside = "BUY" if action== 1 else "SELL"
        Net_Profit = 0
        Active = "Yes"
        exitbool = False
        nest=True if action==5 or action ==6 else False 
        if volume * price > self.funds:
            failed_trade = "Yes"
            exitbool = True
        else:
            failed_trade = "No"

       

        self.trade_log.append(Trade(price, volume, Tradeside, Net_Profit, Active, failed_trade))
        if nest:
            self.further_trades[len(self.trade_log) - 1] = NestedTradeSimulator()
            self.further_trades[len(self.trade_log) - 1].trade_index = len(self.trade_log) - 1
        else:
            self.trade_index = len(self.trade_log) - 1

        self.funds -= volume * price

        if exitbool:
            return [0, self.array_s[self.iters], False]

        list_further = [[key, val] for key, val in self.further_trades.items()]

        for i, bitcoin in enumerate(list_further):
            if bitcoin[1].iters == 0:
                bitcoin[1].state_array(self.array_s)

        action_next = action
        rewardf = lambda price, action: price - self.ltp if action == 1 else self.ltp - price

        reward_bool = [(key, obj.get_state(action_next, next_state, self.iters, self.funds)) for (key, obj) in list_further]
        print(f"reward bool is {reward_bool}")

        reward = rewardf(price, action) * self.vol + sum([suml[0] for _, suml in reward_bool])

        print(f"reward is {reward}")

        for key, reward_data in reward_bool:
            if reward_data[2] == False:
                price_exit = self.array_s[self.iters, 2]
                vols = self.array_s[self.iters, 3]
                vol = self.trade_log[self.further_trades[key].trade_index].Volume
                price = self.trade_log[self.further_trades[key].trade_index].Price
                
                if vol > vols:
                    listvol = []
                    net = vol - vols
                    for i in range(5):
                        if self.trade_log[self.further_trades[key].trade_index].Tradeside == "BUY":
                            listvol.append([net, self.array_s[self.iters + i, 2]])
                            net = self.array_s[self.iters + i + 1, 3] - net
                        else:
                            listvol.append([net, self.array_s[self.iters + i, 4]])
                            net = self.array_s[self.iters + i + 1, 5] - net
                        if net <= 0:
                            break

                    if self.trade_log[self.further_trades[key].trade_index].Tradeside == "BUY":
                        net_profit = (vols * price_exit + sum([n * p for n, p in listvol])) - (vol * price)
                    else:
                        net_profit = vol * price - (vols * price_exit + sum([n * p for n, p in listvol]))

                    self.funds += net_profit
                    self.trade_log[self.further_trades[key].trade_index].Active = "NO"
                    self.trade_log[self.further_trades[key].trade_index].Net_Profit = net_profit
                    del self.further_trades[key]

        quitbool = 1 if self.iters == 150 or reward < self.funds / 2 else 0
        

        if quitbool == 1:
            price_exit = self.array_s[self.iters, 2]
            vols = self.array_s[self.iters, 3]
            vol = self.trade_log[self.trade_index].Volume
            price = self.trade_log[self.trade_index].Price
            
            if vol > vols:
                listvol = []
                net = vol - vols
                for i in range(5):
                    if self.trade_log[self.trade_index].Tradeside == "BUY":
                        listvol.append([net, self.array_s[self.iters + i, 2]])
                        net = self.array_s[self.iters + i + 1, 3] - net
                    else:
                        listvol.append([net, self.array_s[self.iters + i, 4]])
                        net = self.array_s[self.iters + i + 1, 5] - net
                    if net <= 0:
                        break

                if self.trade_log[self.trade_index].Tradeside == "BUY":
                    net_profit = (vols * price_exit + sum([n * p for n, p in listvol])) - (vol * price)
                else:
                    net_profit = vol * price - (vols * price_exit + sum([n * p for n, p in listvol]))

                self.trade_log[self.trade_index].Active = "NO"
                self.trade_log[self.trade_index].Net_Profit = net_profit
                self.funds += net_profit
                self.reset()

        self.ltp = next_state[0]
        next_state = self.array_s[self.iters]
        print(f"next staet is {next_state}")

        return [reward, next_state, quitbool]

    def state_array(self, chunkarr):
        self.array_s = chunkarr
        self.ltp = self.array_s[0, 0]
        return


class NestedTradeSimulator(Bitcoin_simulator):
    def __init__(self,):

        self.iters = 0
        self.trade_index=None
        
    def get_state(self,action,next_state,iters):
        
        self.iters=iters
        quitbool=False if self.iters==100 else True
        rewardf = lambda price, action: price - self.ltp if action == 1 else self.ltp - price
        price = self.array_s[self.iters, 2] if action== 1 else self.array_s[self.iters, 4]
        volume = self.array_s[self.iters, 3] if action == 1 else self.array_s[self.iters, 5]
        reward = rewardf(price, action) * volume 
        quitbool = 1 if self.iters == 150 or reward < self.funds / 4 else 0
        
        self.ltp=next_state[0]
        next_state=self.array_s[self.iters]
      
        
        
        return[reward,next_state,quitbool]
    def state_array(self,arr):
        
        self.array_s=arr
        self.ltp=self.array_s[0,0]


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
