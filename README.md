# pure_NeRF
minimal implementation of NeRF

## train a base NeRF

```sh
python main.py --lrate 5e-4 --lrate_decay 500
```

## train a NeRF with multi-resolution hash encoding

```sh
python main.py --hash
```

training hash mode for 10 minutes:
<p float="center">
  <img src="assets/hash_epoch4k_10min.png" width="30%" />
</p>

training vanilla nerf for 10 minutes:
<p float="center">
  <img src="assets/vanilla_epoch8k_10minutes.png" width="30%" />
</p>

converging speed:
Hash
<p float="center">
  <img src="assets/hash_converging.png" width="30%" />
</p>

vanilla
<p float="center">
  <img src="assets/vanilla_converging1.png" width="30%" />
</p>
