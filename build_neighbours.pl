use strict;
use compute_neighbourhood;

my $verbose = 0;


my $pwd  =  "/home/jsegura/databases/DBD4/benchmark4/structures/Benchmark_3_updated/";
my @in = `cat pdb_list.tsv`;
chomp @in;

foreach my$i(@in){
	print $i."\n";
	my $sufix = $i;
	$sufix =~ s/\.pdb//g;
	my $VD = compute_neighbours->voronoi( $pwd."/".$i, 9 );
        open(FH,">NEIGHBOURS/".$sufix.".vd");
        foreach my$i(keys %{$VD}){
          print FH $i;
          foreach my$j(keys %{ $VD->{$i} }){
            print FH "\t".$j;
          }
          print FH "\n";
        }
        close FH;
}


sub write_nn_features {
	my $sufix = shift;
	my $f_psaia = shift;
	my $seq = shift;
	my $ch_sequence = shift;
	my $nn = shift;
	my $coo = shift;

	my $feature_file = "features/$sufix.nn.tsv";
	open(FH,">$feature_file") or die $feature_file;
	foreach my$i(sort{$a<=>$b}keys %{$f_psaia}){
		print $i."\n"if($verbose);
		my @ND;
		my @NN;
		foreach my$r(@{$nn->{$i}}){
			push @NN, $r->[0];
			push @ND, $r->[1];
		}
		next if(@NN == 0 || @ND == 0);
		print FH $i."\t".$seq->{$i}."\t".join("\t",@{ $f_psaia->{$i} })."\t".join("\t",@NN)."\t".join("\t",@ND)."\t".$coo->{$i}."\n";
	}
	close FH;
}

sub read_psaia {
	my $file = shift;
	my $cmd = "tail --lines=+9 $file";
	my @in = `$cmd`;
	chomp @in;
	my $out;
	my $seq;
	my $ch_seq;
	foreach(@in){
		$_=~s/^(\s+)//;
		my @r = split(/\t+|\s+/,$_);
		my $ch_id = $r[0];
		my $res_id = $r[6];
		next if($res_id=~/([A-Z])$/);
		my $res_type = $r[7];

		#my @aux = @r[8..$#r];
		my @aux = @r[8..12];
		

		my @aux2;
		foreach(@aux){
			my $x = 1/(1+exp(-0.03*($_-25)));
			push @aux2 , substr($x,0,4);
		}
                push @aux2 , substr(1/(1+exp(0.5*($r[$#r]))),0,4);

		#die ($#aux+1),"!!!" unless(@aux == 23);
		$out->{ $res_id.$ch_id } = \@aux2;
		$seq->{ $res_id.$ch_id } = aa_3to1($res_type);
		push @{ $ch_seq->{$ch_id} } ,  [$res_id.$ch_id,aa_3to1($res_type)];
	}
	return ($out,$seq,$ch_seq);
}

sub read_naccess {
	my $file = shift;
	die $file unless(-e $file);
	my @in = `tail --lines=+5 $file | grep -E "RES|HEM"`;
	chomp @in;
	my $out;
	my $seq;
	foreach my$i(@in){
		print $i."\n" if($verbose);
		my $res_type = substr($i,4,3);
		print $res_type."\n"if($verbose);
		my $ch_id = substr($i,8,1);
		$ch_id = '*'if($ch_id eq ' ');
		my $res_id = substr($i,9,4);
		$res_id =~ s/\s//g;
		print $res_id."\n"if($verbose);
		my $flag = substr($i,13,1);
		next if($flag=~/[A-Z]/);
		my $aux = substr($i,14);
		$aux =~ s/^(\s+)//g;
		my @r = split(/\s+/,$aux);
		$out->{ $res_id.$ch_id } = \@r;
		$seq->{ $res_id.$ch_id } = aa_3to1($res_type);
	}
	return ($out,$seq);
}

sub filter_vd {
	my $vd = shift;
	my $rsa = shift;
	my $seq = shift;

	my $out;
	foreach my$i(keys %{$vd}){
		next unless( exists $rsa->{$i} );
		#next unless( exists $rsa->{$i} &&  $rsa->{$i}->[0]>0 );
		foreach my$j(keys %{ $vd->{$i} }){
			$out->{$i}->{$j} = 1 if( exists $rsa->{$j} && $rsa->{$j}->[0]>0 );
		}
	}
	my $f_vd;
	foreach my$i(keys %{$out}){
		$f_vd->{$i} = {A=>0,D=>0,E=>0,G=>0,F=>0,L=>0,S=>0,Y=>0,C=>0,W=>0,P=>0,H=>0,Q=>0,R=>0,I=>0,M=>0,T=>0,N=>0,K=>0,V=>0,Z=>0,ZZ=>0};
		foreach my$j(keys %{ $out->{$i} }){
			$f_vd->{$i}->{ $seq->{$j} }++;
			$f_vd->{$i}->{ ZZ }++;
		}
	}

	my $out2;
	foreach my$i(keys %{$vd}){
		next unless( exists $rsa->{$i}  );
		#next unless( exists $rsa->{$i} &&  $rsa->{$i}->[0]>0 );
		foreach my$j(keys %{ $vd->{$i} }){
			$out2->{$i}->{$j} = 1 if( exists $rsa->{$j} && $rsa->{$j}->[0] == 0 );
		}
	}
	my $f_inner_vd;
	foreach my$i(keys %{$out2}){
		$f_inner_vd->{$i} = {A=>0,D=>0,E=>0,G=>0,F=>0,L=>0,S=>0,Y=>0,C=>0,W=>0,P=>0,H=>0,Q=>0,R=>0,I=>0,M=>0,T=>0,N=>0,K=>0,V=>0,Z=>0,ZZ=>0};
		foreach my$j(keys %{ $out->{$i} }){
			$f_inner_vd->{$i}->{ $seq->{$j} }++;
			$f_inner_vd->{$i}->{ ZZ }++;
		}
	}
	return ($out,$f_vd,$f_inner_vd);
}

sub aa_3to1 {
	my $aa = shift;
	my $__AA = {
		ALA=>'A',TYR=>'Y',MET=>'M',LEU=>'L',CYS=>'C',GLY=>'G',
		ARG=>'R',ASN=>'N',ASP=>'D',GLN=>'Q',GLU=>'E',HIS=>'H',TRP=>'W',
		LYS=>'K',PHE=>'F',PRO=>'P',SER=>'S',THR=>'T',ILE=>'I',VAL=>'V'
	};
	return $__AA->{$aa} if(exists $__AA->{$aa});
	return 'Z';
}
