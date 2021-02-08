@lines = `cat grammar.txt`;
chomp(@lines);

print <<EOF;
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head>
<meta content="text/html; charset=utf-8" http-equiv="content-type" name="Aspen Grammar" keywords="Aspen, Grammar"/>
<title>Aspen Grammar</title>
<body>
<code>
EOF

foreach $f (@lines)
{
    # do quotes first, because they're about to become part of a tag
    $f =~ s/\"/&quot;/g;
    # do caret substitution one first, because we need to change to &gt;, etc.
    $f =~ s/\<([^>]+)\>/&lt;<font color="#884488">$1<\/font>&gt;/g;
    $f =~ s/&quot;([^&]+)&quot;/&quot;<font color="#008800">$1<\/font>&quot;/g;
    #$f =~ s/([^A-Z])([A-Z][A-Z_]+)([^A-Za-z_])/$1<font color="#000088">$2<\/font>$3/g;
    $f =~ s/([^A-Z])([A-Z][A-Z_]+)/$1<font color="#000088">$2<\/font>$3/g;
    #$f =~ s/([A-Z][A-Z]+)/<font color="#000088">$1<\/font>/g;
    $f =~ s/\|/<font color="#600000">\|<\/font>/g;

    # replace spaces at the beginning of the line with non-breaking spaces
    # but note that we have two spaces many places; we can abandon that
    # (html) but need to substract one from the length here.
    $f =~ s{^( +)}{"&nbsp;" x (length($1)-1)}e;
    print "$f<br/>\n";
}

print "</code>\n";
print "</body>\n";

