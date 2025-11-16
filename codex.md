我们的任务是：

1. 模拟Emergencies
   在一个已知number of floor, stairwell placement,rooms, location of exits, occupancy patterns
   的指定layout的building中，在responders.init_position处放置responders.num个responder，
   模拟其按照一定order去sweep不同房间，使得occupants从exits逃生，并确保rooms empty的过程，
   记录整个过程的用时time和rooms被sweep的order。
2. 优化sweep strategy
   使用强化学习算法来调整responders的sweep的order，使得
   a. occupants safe（从exits逃出building）
   b.在action过程中occupants moving towards exits而不是backwards
   c. 确保房间empty没有被遗漏
   d.用时time尽可能少
   给出最优的order，time，和responders.init_position
3. Case Apply
   设定responders.number=2,number of floor=1, layout为：由一个中央走廊（central hallway）
   连接两侧共 6 间办公室（offices）：每侧各有 3 间房间：上方 3 间、下方 3 间；每个房间均有一扇门通向走廊；
   走廊两端各有一个出口（exit），分别通向建筑的两侧；
   在baseline.json对应的layout中Apple步骤1，2，求出最优的order，time，和responders.init_position

例如现在代码给的结果：
{"layout": "baseline.json", "responders": 2, "per_room": 5, "time": 1161, "all_evacuated": true, "room_order": ["R3", "R1", "R4", "R6", "R2", "R5"], "init_positions": [[9, 20], [96, 20]], "exits": [[8, 19], [97, 19]], "evacuated": 30, "real_hms": "00:15:09", "real_minutes": 15.15, "cell_m": 0.5, "speed_solo_mps": 0.8, "speed_escort_mps": 0.6}
≈ Real time: 00:15:09 (cell=0.5m, v_solo=0.8 m/s, v_escort=0.6 m/s)

4. Apply
   基于baseline.json，拓展到layout_A.json,layout_B.json，a=2种额外的layout，设计b种number of floor的building
   环境，给定c种occupants of the rooms的设定，测试n个responders从m种starting position开始sweep的过程，
   记录其最优的order，time，responders.number和responders.init_position，共(a*b*c+1)*n*m次模拟，比较结果，
   得到（a*b*c+1）种结果。其中的1位Case Apply的结果。

- res
  1. layout,number of floor,responders.number,responders.init_position,order(从一楼的哪个exits进入)，time
  2. layout,number of floor,responders.相同的组合中，挑选出number,responders.init_position,order(从一楼的哪个exits进入)里面time最少的那一条作为结果保存到select_result.csv

2. Expand Model
   在现有的模型上添加forg，能见度，速度，死亡的设定，使得模型更加realistic。添加sensor的设定，再次模拟比较
   设置sensor前后的sweep情况。
3. conclusion
   比较并分析结果，发现一些规律性的strategies。

我们对上述Task的第一性理解是：Assisting responders in emergency evacuation by optimizing sweeping strategy.
见下图：

这就引申出三个问题：

1. Question1:What's the criteria of ALL-CLEAR?
2. Question2:How to arrange the Sweep Strategy?
3. Question3:When Does a Sweep Take Its Total Time?

q1的方案是：Full Coverage, No Presence,Double-check confirmation,三个latex数学公式描述的条件
q2的方案是：先用贪心算法跑通流程，给出一次模拟的结果和批量模拟的结果，然后用RL探索最佳的Order。
q3的方案是，用2D可视化（目前代码已经实现了一楼的模拟，还没实现n楼的模拟），和Minecraft（不在该项目中实现，但是与该项目共用configs）来模拟并计时

给我讲讲现在repo里的代码。基本都是gpt写的，我不了解，也无从修改和写论文

---

我需要对repo的代码有掌控感。但是大段的代码其实不利于实现这一点，因此整个代码的逻辑需要被改变。main.py应该是高度抽象而简介的逻辑，几乎约等于流程描述。而具体的代码细节再utils.py这样的代码中作为工具函数被实现。我1.想知道目前的代码和我的设计路线有哪些差距 2. 我的设计路线应该如何完善 3. 如何把代码修改成我可以掌握的架构。

---

---

你觉得现在的make命令的安排怎么样？我对目前基于贪心的det det-gif很满意，希望把强化学习也集成进去，最好通过一个algorithm=greed/ppo的选项，直接make run就能自动抛出结果和生成gif。然后通过make batch批量生产下面这样的批量结果用于分析：layout,floors,per_room_occ,responders,exit_combo,makespan_hms,room_clear_order,exit_combo_id,makespan_s,responder_orders,start_positions
BASELINE,1,5,1,L,00:10:16,R3_F1->R0_F1->R1_F1->R4_F1->R5_F1->R2_F1,0,615.6098348886976,F1:R3_F1->R0_F1->R1_F1->R4_F1->R5_F1->R2_F1,F1:E_L

---

1. 在make batch上也实现sweep.gif
2. 通过FLOORS=2 make run来实现2层的模拟
3. batch_runs也放在output下
4. 目前的det_baseline.json改名为baseline_greedy_result.json，打印真正的log到run.log纯文本中，首先打印所有初始化的参数，然后打印每个action，每个移动，每个room的check的细节，最后打印最终的结果summary，并且保存结果到baseline_greedy_result.json中
5. ppo的结果保存为baseline_ppo_result.json，log保存为run_ppo.log

---

1. 真正把 --floors 做成“多层 3D 网格 + 楼梯竖井”的实现
2. 每一次make run的output不要直接放在output下，而是生成一个以参数取值命名的子文件夹，里面放结果和log
3. batch的结果也放在一个以参数取值命名的子文件夹
4. 目前的ppo实现还不完善，先把贪心的结果完善
5. 在make batch上也实现sweep.gif作为一个选项，默认关闭。

---

1. make batch的结果里只有time，没有"time": 2153,
   "real_hms": "00:25:57",
   "real_minutes": 25.95,
   补上
2. 现在跑的太慢了。加上checkpoint再det_batch.jsonl里检查参数的组合，对于没有跑过结果的组合，开多线程跑
