### Testing for Adversarial Attacks

* Models are referenced from https://github.com/osmr/imgclsmob
* Code for transfer attack is referenced from https://github.com/Trustworthy-AI-Group/TransferAttack


To run the code, simply run `main.py` with optional arguments.  

* Example : running `FGSM` on `resnet110_cifar100` with evaluation

```
python3 main.py --input_dir ./data --output_dir ./results --attack fgsm --model resnet110_cifar100
python3 main.py --eval --input_dir ./data --output_dir ./results --attack fgsm --model resnet110_cifar100
```

* The arguments are listed below :

```
-h, --help            show this help message and exit
-e, --eval            attack/evluation
--attack {fgsm,ifgsm,mifgsm,nifgsm,pifgsm,vmifgsm,vnifgsm,emifgsm,ifgssm,vaifgsm,aifgtm,rap,gifgsm,pcifgsm,iefgsm,dta,gra,pgn,smifgrm,dim,tim,sim,atta,admix,dem,odi,ssm,aitl,maskblock,sia,stm,bsr,decowa,l2t,tap,ila,potrip,fia,yaila,logit,trap,naa,rpa,taig,fmaa,cfm,logit_margin,ilpd,fft,ir,sgm,iaa,dsm,mta,mup,bpa,dhf,pna_patchout,sapr,tgr,ghost,setr,ags} 
                      the attack algorithm
--epoch EPOCH         the iterations for updating the adversarial patch
--batchsize BATCHSIZE
                      the bacth size
--eps EPS             the stepsize to update the perturbation
--alpha ALPHA         the stepsize to update the perturbation
--momentum MOMENTUM   the decay factor for momentum based attack
--model MODEL         the source surrogate model
--ensemble            enable ensemble attack
--input_dir INPUT_DIR
                      the path for custom benign images, default: untargeted attack data
--output_dir OUTPUT_DIR
                      the path to store the adversarial patches
--targeted            targeted attack
--GPU_ID GPU_ID
```

* `run.sh` can also be manipulated to achieve autonomous execution
