#run from inside the Captured folder!
#!/bin/sh
echo id ist $1
mkdir ../aligned
mkdir ../aligned/$1
cd $1
filelist=`ls *.jpg | sort -n -t - -k 2`
if [ "$2" != "" ]; then
	echo "reverse file order"
	filelist=`ls *.jpg | sort -n -t - -k 2 -r`
fi
echo $filelist
/Applications/hugin/HuginTools/align_image_stack -m -a ../../aligned/$1/$1- $filelist -v
cd ..
../build/compress -compr 95 -indir ../aligned/$1 -outdir ../samples/$1 -type jpg -color 1 -anydepth 1 -debug 0

../build/vdff -dir ../samples/$1 -export ../samples/results/$1.png