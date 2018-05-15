use strict;

my  @in = `cat ../437_dimers_list.merge.tsv`;
chomp @in;

foreach(@in){
  my @r = split(/\t/,$_);
  my $cmd = "rsync -rv rsanchez\@campins:/home/rsanchez/Tesis/pssmDB/data/databaseUpdate2/uniref90/pssmInZipWithStructId/$r[0]_$r[1]_3.pssm.zip .";
  system($cmd);
  print $cmd."\n";
  my $cmd = "rsync -rv rsanchez\@campins:/home/rsanchez/Tesis/pssmDB/data/databaseUpdate2/uniref90/pssmInZipWithStructId/$r[0]_$r[2]_3.pssm.zip .";
  print $cmd."\n";
  system($cmd);
}
