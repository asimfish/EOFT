

## uniform_WO without gurobi
python3 main.py  --data_type 'uniform_WO' --data_pah '/your_path/data/uniform/node_50_400_50_256.json' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4    

## uniform_WO with gurobi
python3 main.py  --data_type 'uniform_WO' --data_pah '/your_path/data/uniform/node_50_400_50_256.json' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4   --need_gurobi  


## uniform_EN without gurobi
python main.py --data_type 'uniform_EN' --data_pah '/your_path/data/uniform/node_50_400_50_256.json' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4    --Node_C 0.5 --Edge_C 0.5  

## uniform_EN with gurobi
python main.py --data_type 'uniform_EN' --data_pah '/your_path/data/uniform/node_50_400_50_256.json' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4    --Node_C 0.5 --Edge_C 0.5 --need_gurobi 

## uniform_EN without gurobi
python main.py --data_type 'netgen' --data_pah '/your_path/data/uniform/node_50_400_50' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4    --Node_C 0.5 --Edge_C 0.5  

## uniform_EN with gurobi
python main.py --data_type 'netgen' --data_pah '/your_path/data/uniform/node_50_400_50' --seed 1234 -a 50 -b 400 -c 50  --err 1e-5   --batch_size 256 --iters 10000    --eps 1e-3 --d0 1e-5   --iters 150000 --err 1e-4    --Node_C 0.5 --Edge_C 0.5 --need_gurobi 

