all:
	@

train:
	python3 train.py

board:
	tensorboard --logdir=./train_log

clean:
	rm -rf train_log
