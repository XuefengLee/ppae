#/usr/bin/python3                                                                                                                        

# mnist                                                                                                                                  
#python3 main.py --dataroot /home/user/Desktop/datasets  --dataset mnist -n_z 8 --batch_size 100 --device 0 --save_dir mnist --ld 1  > mnist/log.txt                                                                                                           #python3 main.py --dataroot /home/user/Desktop/datasets  --dataset mnist -n_z 8 --batch_size 100 --device 1 --save_dir mnist/2 --ld 10  > mnist/2/log.txt


python3 main.py --dataroot /home/user/Desktop/datasets  --dataset celeba -n_z 64 --batch_size 100 --device 0 --save_dir celeba --alpha=0.45 --scale=0.64 --ld 0.1

python3 main.py --dataroot /home/user/Desktop/datasets  --dataset celeba -n_z 64 --batch_size 100 --device 0 --save_dir celeba --alpha=0.0 --scale=0.7071 --ld 0.1

python3 main.py --dataroot /home/user/Desktop/datasets  --dataset celeba -n_z 64 --batch_size 100 --device 0 --save_dir celeba --alpha=0.45 --adaptive --ld 0.000001

python3 main.py --dataroot /home/user/Desktop/datasets  --dataset celeba -n_z 64 --batch_size 100 --device 0 --save_dir celeba --alpha=0.0 --adaptive --ld 0.000001




