i=1
F=NSLKDD
while [ $i -le 36 ]
do
    cp -vf NSLKDD_checkpoint.pth NSLKDD_${i}_checkpoint.pth
    let i+=1
done
