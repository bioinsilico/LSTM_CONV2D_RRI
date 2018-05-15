use strict;

my @in = `cat dimers_117_list.part.a*`;
chomp @in;

foreach(@in){
  print $_."\n" if(!-e "results/".$_.".tsv");
}
