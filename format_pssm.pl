use strict;

my @in = `ls *zip`;
chomp @in;

foreach(@in){
  print $_."\n";
  my $ch;
  if(/^(\w{4}_)(\w)(_)/){
    $ch = $2;
  }
  my $name = $_;
  $name =~ s/\.zip//g;
  my @FH = `zcat $_`;
  chomp @FH;
  my $n = 0;
  open(FH,">$name");
  print FH "chainId seqIndex structResId resName pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm pssm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm psfm score score\n";
  foreach my$L(@FH){
    if($L=~/^(\s+)(\d+)/){
      my @r = split(/\s+/,$L);
      print FH $ch." ".$n." ".$r[1]." ".$r[2]." ".join(" ",@r[3..$#r])."\n";
      $n++;
    }    
  }
  close FH;
}
