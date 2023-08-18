function decode(facecode::UInt8)
    # Numbering of quadrant in facecode
    #   2-----3-----3
    #   |           |
    #   |y          |
    #   0^          1
    #   ||          |
    #   |+-->x      |
    #   0-----2-----1
    corner = facecode & 0x3
    xfaceishanging = (facecode >> 0x2) & 0x1 == 0x1
    yfaceishanging = (facecode >> 0x3) & 0x1 == 0x1

    hangingface = (
        ((corner == 0x0) && xfaceishanging) ? 0x1 :
        ((corner == 0x2) && xfaceishanging) ? 0x2 : 0x0,
        ((corner == 0x1) && xfaceishanging) ? 0x1 :
        ((corner == 0x3) && xfaceishanging) ? 0x2 : 0x0,
        ((corner == 0x0) && yfaceishanging) ? 0x1 :
        ((corner == 0x1) && yfaceishanging) ? 0x2 : 0x0,
        ((corner == 0x2) && yfaceishanging) ? 0x1 :
        ((corner == 0x3) && yfaceishanging) ? 0x2 : 0x0,
    )

    return hangingface
end

function decode(facecode::UInt16)
    # Numbering of octant in facecode
    #               6-----3-----7
    #               |           |
    #               |z          |
    #              10^    3    11
    #               ||          |
    #               |+-->x      |
    #   6----10-----2-----1-----3----11-----7-----3-----6
    #   |           |           |           |           |
    #   |          y|y          |y          |          y|
    #   6     0    ^4^    4     5^    1     7     5    ^6
    #   |          |||          ||          |          ||
    #   |      z<--+|+-->x      |+-->z      |      x<--+|
    #   4-----8-----0-----0-----1-----9-----5-----2-----4
    #               |+-->x      |
    #               ||          |
    #               8v    2     9
    #               |z          |
    #               |           |
    #               4-----2-----5
    xfaceishanging = (facecode >> 0x3) & 0x1 == 0x1
    yfaceishanging = (facecode >> 0x4) & 0x1 == 0x1
    zfaceishanging = (facecode >> 0x5) & 0x1 == 0x1

    xedgeishanging = (facecode >> 0x6) & 0x1 == 0x1
    yedgeishanging = (facecode >> 0x7) & 0x1 == 0x1
    zedgeishanging = (facecode >> 0x8) & 0x1 == 0x1

    corner = facecode & 0x0007
    xcornershift = (corner >> 0x0) & 0x1 == 0x1
    ycornershift = (corner >> 0x1) & 0x1 == 0x1
    zcornershift = (corner >> 0x2) & 0x1 == 0x1

    hangingface = (
        ((corner === 0x0000) && xfaceishanging) ? 0x1 :
        ((corner === 0x0002) && xfaceishanging) ? 0x2 :
        ((corner === 0x0004) && xfaceishanging) ? 0x3 :
        ((corner === 0x0006) && xfaceishanging) ? 0x4 : 0x0,
        ((corner === 0x0001) && xfaceishanging) ? 0x1 :
        ((corner === 0x0003) && xfaceishanging) ? 0x2 :
        ((corner === 0x0005) && xfaceishanging) ? 0x3 :
        ((corner === 0x0007) && xfaceishanging) ? 0x4 : 0x0,
        ((corner === 0x0000) && yfaceishanging) ? 0x1 :
        ((corner === 0x0001) && yfaceishanging) ? 0x2 :
        ((corner === 0x0004) && yfaceishanging) ? 0x3 :
        ((corner === 0x0005) && yfaceishanging) ? 0x4 : 0x0,
        ((corner === 0x0002) && yfaceishanging) ? 0x1 :
        ((corner === 0x0003) && yfaceishanging) ? 0x2 :
        ((corner === 0x0006) && yfaceishanging) ? 0x3 :
        ((corner === 0x0007) && yfaceishanging) ? 0x4 : 0x0,
        ((corner === 0x0000) && zfaceishanging) ? 0x1 :
        ((corner === 0x0001) && zfaceishanging) ? 0x2 :
        ((corner === 0x0002) && zfaceishanging) ? 0x3 :
        ((corner === 0x0003) && zfaceishanging) ? 0x4 : 0x0,
        ((corner === 0x0004) && zfaceishanging) ? 0x1 :
        ((corner === 0x0005) && zfaceishanging) ? 0x2 :
        ((corner === 0x0006) && zfaceishanging) ? 0x3 :
        ((corner === 0x0007) && zfaceishanging) ? 0x4 : 0x0,
    )

    onhangingface = (
        hangingface[3] > 0x0 || hangingface[5] > 0x0,
        hangingface[4] > 0x0 || hangingface[5] > 0x0,
        hangingface[3] > 0x0 || hangingface[6] > 0x0,
        hangingface[4] > 0x0 || hangingface[6] > 0x0,
        hangingface[1] > 0x0 || hangingface[5] > 0x0,
        hangingface[2] > 0x0 || hangingface[5] > 0x0,
        hangingface[1] > 0x0 || hangingface[6] > 0x0,
        hangingface[2] > 0x0 || hangingface[6] > 0x0,
        hangingface[1] > 0x0 || hangingface[3] > 0x0,
        hangingface[2] > 0x0 || hangingface[3] > 0x0,
        hangingface[1] > 0x0 || hangingface[4] > 0x0,
        hangingface[2] > 0x0 || hangingface[4] > 0x0,
    )

    onhangingedge = (
        xedgeishanging && (corner === 0x0000 || corner === 0x0001),
        xedgeishanging && (corner === 0x0002 || corner === 0x0003),
        xedgeishanging && (corner === 0x0004 || corner === 0x0005),
        xedgeishanging && (corner === 0x0006 || corner === 0x0007),
        yedgeishanging && (corner === 0x0000 || corner === 0x0002),
        yedgeishanging && (corner === 0x0001 || corner === 0x0003),
        yedgeishanging && (corner === 0x0004 || corner === 0x0006),
        yedgeishanging && (corner === 0x0005 || corner === 0x0007),
        zedgeishanging && (corner === 0x0000 || corner === 0x0004),
        zedgeishanging && (corner === 0x0001 || corner === 0x0005),
        zedgeishanging && (corner === 0x0002 || corner === 0x0006),
        zedgeishanging && (corner === 0x0003 || corner === 0x0007),
    )

    shift = (
        xcornershift,
        xcornershift,
        xcornershift,
        xcornershift,
        ycornershift,
        ycornershift,
        ycornershift,
        ycornershift,
        zcornershift,
        zcornershift,
        zcornershift,
        zcornershift,
    )

    hangingedge = ntuple(Val(12)) do n
        onhangingedge[n] ? ((onhangingface[n] ? 0x3 : 0x1) + shift[n]) :
        (onhangingface[n] ? 0x5 : 0x0)
    end

    return (hangingface, hangingedge)
end
