import csv

log_loss = open("./a.csv", "w")
logger = csv.writer(log_loss, lineterminator="\n")
carrier = [123685.564321]
logger.writerow(carrier)
k = input("end?")
