@subst_from = ();
@subst_to = ();

push @subst_from, "TINT";
push @subst_to, "<integer>";

push @subst_from, "TREAL";
push @subst_to, "<real>";

push @subst_from, "TIDENT";
push @subst_to, "<identifier>";

push @subst_from, "TSTRING";
push @subst_to, "<string>";

push @subst_from, "optionalstring";
push @subst_to, "(<string> | <empty>)";

push @subst_from, "optionalident";
push @subst_to, "(<identifier> | <empty>)";

push @subst_from, "/\\* empty \\*/";
push @subst_to, "<empty>";

push @subst_from, "TKW_10POWER";
push @subst_to, "<si_prefix>";


@grammar = ();
open(GF, "<../aspen/parser/AspenParser.output") || die "Can't find ../aspen/AspenParser.output; please build Aspen before generating the grammar.\n";
$mode = 0;
while (my $line = <GF>)
{
    chomp($line);
    if ($mode == 0 and $line eq "Grammar")
    {
        $mode = 1;
        next;
    }
    if ($mode == 1 and $line eq "Terminals, with rules where they appear")
    {
        last;
    }
    if ($mode == 1)
    {
        $line = substr($line,6);
        if ($line =~ m/^optional\w+/)
        {
            # this rule is just to to a (x | <empty>);
            # don't add the rule, and do the right BNF substitution

            ($f = $line) =~ s/^(optional\w+).*$/$1/;
            ($t = $line) =~ s/^optional(\w+).*$/$1/;
            # we'll also uppercase the latter piece and assume it's 
            # a nonterminal (because we have explicit substitutions
            # for optional terminals like string.
            $t = uc($t);
            $t = "($t | <empty>)";
            push @subst_from, $f;
            push @subst_to, $t;
            # skip until next rule
            while ($line = <GF>)
            {
                chomp($line);
                last if $line eq "";
            }
        }
        else
        {
            $line =~ s/^(\w+): /$1  =  /;
            $line =~ s/^(\s+)\| /$1  |  /;
            # add uppercasing to nonterminals since keywords are terminal
            if ($line =~ m/^\w+/)
            {
                ($f = $line) =~ s/^(\w+) .*$/$1/;
                $t = uc($f);
                #print "from=$f to=$t\n";
                push @subst_from, " $f ";
                push @subst_to, " $t ";
                push @subst_from, "^$f ";
                push @subst_to, "$t ";
                push @subst_from, " $f\$";
                push @subst_to, " $t";
            }
            push @grammar, $line;
        }
    }
}
close(GF);

#print join "\n", @grammar;


open(TF, "<../aspen/parser/AspenTokens.l") || die "Can't find ../aspen/parser/AspenTokens.l; are you running from the proper directory?\n";
while (my $line = <TF>)
{
    chomp($line);
    if ($line =~ m/^\w+(\|\w+)?\s+return TOKEN\(TKW_\w+\);$/)
    {
        #print "found match1: $line\n";
        ($f = $line) =~ s/^\w+(\|\w+)?\s+return TOKEN\((TKW_\w+)\);$/$2 /;
        ($t = $line) =~ s/^(\w+)(\|\w+)?\s+return TOKEN\(TKW_\w+\);$/\"$1\"  /;
        #print "from=$f to=$t\n";
        push @subst_from, $f;
        push @subst_to, $t;
        ($f = $line) =~ s/^\w+(\|\w+)?\s+return TOKEN\((TKW_\w+)\);$/$2\$/;
        ($t = $line) =~ s/^(\w+)(\|\w+)?\s+return TOKEN\(TKW_\w+\);$/\"$1\" /;
        #print "from=$f to=$t\n";
        push @subst_from, $f;
        push @subst_to, $t;
    }
    if ($line =~ m/^\"\S+\"\s+return TOKEN\(\w+\);$/)
    {
        #print "found match2: $line\n";
        ($f = $line) =~ s/^\"\S+\"\s+return TOKEN\((\w+)\);$/$1/;
        ($t = $line) =~ s/^(\"\S+\")\s+return TOKEN\(\w+\);$/$1 /;
        #print "from=$f to=$t\n";
        push @subst_from, $f;
        push @subst_to, $t;
    }
}
close(TF);

$n = scalar(@subst_from);
foreach $f (@grammar)
{
    for ($i=0; $i<$n; $i++)
    {
        $from = $subst_from[$i];
        $to = $subst_to[$i];
        $f =~ s/$from/$to/g;
    }
}

print join "\n", @grammar;
