use strict;

my @in = `ls dimers_bipspi/*pssm`;
chomp @in;

foreach(@in){
  my $n = `tail -1 $_ | cut -d" " -f2`; 
  chomp $n;
  print $_."\t".$n."\n";
}
