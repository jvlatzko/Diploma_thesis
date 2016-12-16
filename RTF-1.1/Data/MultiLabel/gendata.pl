#!/usr/bin/perl -W
use Data::Dumper;

my @binary = ( [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1] );

my @labels = ( [2],
               [1],
               [1, 2],
               [0],
               [0, 2],
               [0, 1],
               [0, 1, 2] );

my @probs = ( 0.05,
              0.05,
              0.3,
              0.2,
              0.1,
              0.1,
              0.2 );

my @acc_probs = ( 0.05,
                  0.1,
                  0.4,
                  0.6,
                  0.7,
                  0.8,
                  1.0 );

sub outcome
{
    my $num = shift;

    for( my $i = 0; $i < 7; ++$i )
    {
	if( $num <= $acc_probs[$i] )
	{
	    return $i;
	}
    }
}
                   
for ( my $n = 0; $n < 100; ++$n )
{
    my $out = outcome(rand());
    my $bin = $binary[$out];
    my $lab = $labels[$out];

    print STDOUT join(',', @$lab) . " ";

    my $n0 = 2.0 * (0.5 - rand());
    my $n1 = 2.0 * (0.5 - rand());
    my $n2 = 2.0 * (0.5 - rand());

    print STDOUT "1:" . ($bin->[0]-$n0) . " 2:" . ($bin->[1]-$n1) . " 3:" . ($bin->[2]-$n2) . "\n"; 
}

exit 0;
