#/usr/bin/python3                                                                                                                        

# mnist                                                                                                                                  
#python3 main.py --dataroot /home/user/Desktop/datasets  --dataset mnist -n_z 8 --batch_size 100 --device 0 --save_dir mnist --ld 1  > mnist/log.txt                                                                                                                             

python3 main.py --dataroot /home/user/Desktop/datasets  --dataset mnist -n_z 8 --batch_size 100 --device 1 --save_dir mnist/2 --ld 10  > mnist/2/log.txt

#python3 main.py --dataroot /home/user/Desktop/datasets  --dataset celeba -n_z 64 --batch_size 100 --device 0 --save_dir celeba --ld 1   
