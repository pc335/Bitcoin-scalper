BTC Orderbook Trading Simulator
Project Overview
This project is a Bitcoin orderbook-based trading simulator where prices have been normalized relative to the opening price minus 2. It uses custom preprocessing on orderbook data to extract key features and simulate short-term trading behavior. The code is structured with nested trade handling and a simple DQN (Deep Q-Network) training loop.

Preprocessing
The data preprocessing does a few important things:

First, we load in all the columns from the orderbook data.

Then, we create a Market Depth Ratio (MDR) by dividing the total bid volume (top 5 levels) by the total ask volume (top 5 levels).

After that, we generate an MDR signal where it gives a 1 if bid side is stronger, -1 if ask side is stronger.

We also calculate a Cumulative MDR by adding the current MDR to the previous one, and generate a signal for that too.

We compute an Order Flow Imbalance (OFI) using the change in top bid and ask volumes over time.

OFI signal is created similarly, where positive imbalance gives 1, negative gives -1, and neutral gives 0.

We calculate a Volume-Weighted Mid Price (VQMP) using top of book prices and volumes.

Delta is the difference in trade timestamps, basically the time between trades.

We also detect high volatility situations if trades occur very fast (under 10ms).

Finally, we calculate spread between best ask and best bid.
Prices have been normalized relative to the opening price minus 2. So during inference, 
when real-time predictions are made, we must normalize incoming prices using the same logic 
 take the current opening price minus 2 and adjust accordingly to match training data behavior.


 Trade Handling (Nested Structure)
The trading simulator is set up to handle nested trades.
This means that:

If a position is already open, we can either stick, hit buy/sell, or get out (close the position).

If no position is open, we can hit buy, hit sell, or stick.

The trade actions are handled inside a clean structure where the simulator checks the current position, updates it based on actions, and calculates the profit or loss at exit.

Rewards are based on whether price moves in favor (+1) or against (-1) the position.
If the action is "stick", the reward is 0.

Training Loop
The training loop uses a basic Deep Q-Network (DQN) setup:
The agent takes the current market state and selects an action.

The action is applied in the simulator, which returns the next state and a reward.

The experience (state, action, reward, next state, done) is stored in memory.

We sample random minibatches from memory to train the model and update Q-values.

Over time, the agent learns which actions to take to maximize profits based on orderbook signals.

We also use a decaying epsilon strategy to balance exploration and exploitation during training.



