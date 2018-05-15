use strict;

my $results = $ARGV[0] or die "!!!!!!";
my @in = `ls $results/ | grep tsv`;
chomp @in;

my $epoch;
foreach(@in){
  my @R = `cat $results/$_`;
  chomp @R;
  foreach(@R){
    my @V = split(/ /,$_);
    my @W = split(/:|=/,$V[5]);
    #print $V[0]."\t".$W[1]."\t".$W[2]."\n";
    push @{ $epoch->{$V[0]} } , $W[1] if($W[1] !~ /N/i);
    push @{ $epoch->{$V[0]} } , $W[2] if($W[2] !~ /N/i);
  }
}

foreach my$e(sort{$a<=>$b}keys %{$epoch}){
  my $avg = avg(@{ $epoch->{$e} });
  print $e."\t".$avg."\n";
}

sub avg {
  my @X = @_;
  my $N = @X;
  my $out = 0;
  foreach(@X){
    $out += $_;
  }
  return $out/$N;
}
