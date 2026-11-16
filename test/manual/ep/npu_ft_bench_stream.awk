# Preserve wire data and timestamp it before JSON decoding; no per-token process.
BEGIN { FS = "\t" }
{
    getline clock < clock_file
    close(clock_file)
    split(clock, fields, " ")
    stamp = fields[1]
    if (stamp == "") exit 2
    print $0 >> (dir "/stream.sse")
    print stamp "\t" $0 >> (dir "/wire.tsv")
    fflush()
    if (substr($0, 1, 6) == "data: ") {
        print stamp "\t" substr($0, 7)
        fflush()
    } else if (substr($0, 1, 16) == "__HTTP_STATUS__:") {
        print substr($0, 17) > (dir "/http.txt")
        close(dir "/http.txt")
    }
}
