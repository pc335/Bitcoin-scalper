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
        self.trade_index = []
        self.trade_side_dict={0:"Sell",1:"Buy",2:"Stay",3:"Quit",4:"FURTHER BUY",5:"FURTHER SELL"}
       
      

    def reset(self):
        self.ltp = 0
        self.vol = 0
        self.further_trades = {}
        self.trade_log = []
        self.trade_index =[]
        self.iters = 0
        

    def get_state(self, action, next_state, nest=False):
       
        quitbool=1 if action==4 else 0
        self.iters += 1
 
        self.ltp=self.array_s[self.iters, 0]
        if action in [0,1]:
            price = self.array_s[self.iters, 4] if action== 1 else self.array_s[self.iters, 2].item()
            volume = self.array_s[self.iters, 5] if action == 1 else self.array_s[self.iters, 3].item()
            self.vol=volume
            Tradeside = self.trade_side_dict[action]
            Net_Profit = 0
            Active = "Yes"
            exitbool = False
            if volume * price > self.funds:
                failed_trade = "Yes"
                exitbool = True
          
            else:
                failed_trade = "No"
            self.trade_log.append(Trade(price, volume, Tradeside, Net_Profit, Active, failed_trade))
            self.trade_index.append(len(self.trade_log) - 1)
        nest=True if action==4 or action ==5 else False 
        if action in [2,3,4,5]:
        
            if action==2 or 3:
                Net_Profit=0
                Tradeside=self.trade_log[self.trade_index[0]].Tradeside
                volume = self.trade_log[self.trade_index[0]].Volume
                price = self.trade_log[self.trade_index[0]].Price
                Active="Yes" if action==2 else "No"
                failed_trade="No"
            else:
                price = self.array_s[self.iters, 4] if action== 1 else self.array_s[self.iters, 2].item()
                volume = self.array_s[self.iters, 5] if action == 1 else self.array_s[self.iters, 3].item()
                self.vol=volume
                Tradeside = self.trade_side_dict[action]
                Net_Profit = 0
                Active = "Yes"
                exitbool = False
            if volume * price > self.funds:
                failed_trade = "Yes"
                exitbool = True
          
            else:
                failed_trade = "No"
            
            exitbool=False
       
        
        self.trade_log.append(Trade(price, volume, Tradeside, Net_Profit, Active, failed_trade))
        
        self.trade_index.append(len(self.trade_log) - 1)
       
        if nest:
            self.further_trades[len(self.trade_log) - 1] = NestedTradeSimulator()
            self.further_trades[len(self.trade_log) - 1].trade_index = len(self.trade_log) - 1
       
            
           

        self.funds -= volume * price

        if exitbool:
            return [0, self.array_s[self.iters], False]

        list_further = [[key, val] for key, val in self.further_trades.items()]

        for i, bitcoin in enumerate(list_further):
            if bitcoin[1].iters == 0:
                bitcoin[1].state_array(self.array_s)

        action_next = action
       
        reward_bool = [(key, obj.get_state(action_next, next_state, self.iters, self.funds)) for (key, obj) in list_further]
        if action==0 or action==1:
            rewardf = lambda price, action: price - self.ltp if action == 1 else self.ltp - price

            reward1=rewardf(price, action)
            
       
            reward = reward1 * self.vol -self.ltp*self.vol+ sum([suml[0] for _, suml in reward_bool])
        
        elif action in [2,3,4,5] :
            print(f"self.trad log{self.trade_log}")
            trades=self.trade_log[self.trade_index[0]].Tradeside
            vol = self.trade_log[self.trade_index[0]].Volume
            price = self.trade_log[self.trade_index[0]].Price
            rewardf = lambda price, action: price - self.ltp if action == "BUY" else self.ltp - price
            
       
            reward1=rewardf(price, action)
        
            reward = reward1 * self.vol -self.ltp*self.vol+ sum([suml[0] for _, suml in reward_bool])
       
          
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

        quitbool = 1 if self.iters == 150 or action==3 or reward < -self.funds / 2 else 0
        

        if quitbool == 1:
            price_exit = self.array_s[self.iters, 2]
            vols = self.array_s[self.iters, 3]
            vol = self.trade_log[self.trade_index[0]].Volume
            price = self.trade_log[self.trade_index[0]].Price
            
            if vol > vols:
                listvol = []
                net = vol - vols
                for i in range(5):
                    if self.trade_log[self.trade_index[0]].Tradeside == "BUY":
                        listvol.append([net, self.array_s[self.iters + i, 2]])
                        net = self.array_s[self.iters + i + 1, 3] - net
                    else:
                        listvol.append([net, self.array_s[self.iters + i, 4]])
                        net = self.array_s[self.iters + i + 1, 5] - net
                    if net <= 0:
                        break

                if self.trade_log[self.trade_index[0]].Tradeside == "BUY":
                    net_profit = (vols * price_exit + sum([n * p for n, p in listvol])) - (vol * price)
                   
                else:
                    net_profit = vol * price - (vols * price_exit + sum([n * p for n, p in listvol]))
                 
                self.trade_log[self.trade_index[0]].Active = "NO"
                self.trade_log[self.trade_index[0]].Net_Profit = net_profit
                self.funds += net_profit
                self.reset()

        self.ltp = next_state[0]
        next_state = self.array_s[self.iters]
        

        return [reward, next_state, quitbool]

    def state_array(self, chunkarr):
        self.array_s = chunkarr
        self.ltp = self.array_s[0, 0]
        return


class NestedTradeSimulator(Bitcoin_simulator):
    def __init__(self,):

        self.iters = 0
        self.trade_index=None
        self.vol=0
        self.tradeside=None
        self.intialized=0
        
    def get_state(self,action,next_state,iters,funds):
        if self.intialized==0:
            self.tradeside="BUY" if action==4 else "SELL"
        self.iters=iters
        self.intialized+=1
        
        rewardf = lambda price, action: price - self.ltp if action == 4 else self.ltp - price
       

        price = self.array_s[self.iters, 2].item() if self.tradeside=="BUY" else self.array_s[self.iters, 4].item()
        self.vol = self.array_s[self.iters, 3].item() if self.tradeside =="BUY" else self.array_s[self.iters, 5].item()
 
       
        reward1 = rewardf(price, self.tradeside) 
        reward1 = reward1.item() if isinstance(reward1, torch.Tensor) else reward1
        reward=reward1*self.vol
       
        quitbool = 1 if self.iters == 150 or reward < -funds / 4 else 0
        
        self.ltp=next_state[0]
        next_state=self.array_s[self.iters]
      
        
        
        return[reward,next_state,quitbool]
    def state_array(self,arr):
        
        self.array_s=arr
        self.ltp=self.array_s[0,0]
