<!-- omit in TOC -->

# FSD2JSON

Convert Eve Echoes Static Data FSD Files to JSON for easier inspection.

Converted files can be found [here](http://eve-echoes-data.s3-website.eu-central-1.amazonaws.com/) (last updated for 1.5.6 on 08/29/2020)

## Installing

All you have to do to build it is clone it an run on of the following:

```
cargo install --path .
```

## Usage

Example:

```
fsd2json test.sd
```

This will convert the given `test.sd` file to json and output the result to `test.json`

More info on how to use it can be found in the help section.
`fsd2json --help`
