### The entire directory structure is as follows: 

```
├── demo                   //Jupyter DEMO 
├── doc                    //Tutorial
│   └── install.md               
├── lbitcls                //Core Module
│   ├── apis               //Train,Test,Inference API
│   ├── core               //Eval, Fp16, and etc
│   ├── datasets           //Dataset and Dataloader
│   ├── __init__.py         
│   ├── models             //Models: Backbone, Neck, Loss, Head
│   ├── ops                //Cutomer Operation
│   ├── __pycache__       
│   ├── utils              //Tools
│   ├── VERSION            //Version Info
│   └── version.py
├── README.md
├── requirements           //Requirements
│   ├── build.txt
│   ├── docs.txt
│   ├── optional.txt
│   ├── readthedocs.txt
│   ├── runtime.txt
│   └── tests.txt
├── setup.py               //Install Python Script
├── thirdparty             //Thirdparty
│   └── configs            //Running Configure
├── tools
│   ├── dist_train.sh      //Distribution Training On Brain++
│   └── train.py           //Starting Training Script
└── work_dirs              //Your Working directory
    └── DSQ
````