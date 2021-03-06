/*
 ** TailwindCSS Configuration File
 **
 ** Docs: https://tailwindcss.com/docs/configuration
 ** Default: https://github.com/tailwindcss/tailwindcss/blob/master/stubs/defaultConfig.stub.js
 */
module.exports = {
  theme: {
    extend: (theme) => ({
      fontFamily: {
        sans: ['Inter', ...theme('fontFamily.sans')],
      },
    }),
    typography: (theme) => ({
      default: {
        css: {
          a: {
            color: theme('colors.indigo.500'),
            '&:hover': {
              color: theme('colors.indigo.600'),
            },
          },
          code: {
            backgroundColor: theme('colors.gray.100'),
          },
          'pre code': {
            fontSize: theme('fontSize.base'),
          },
        },
      },
    }),
  },
  variants: {},
  plugins: [require('@tailwindcss/typography')],
  purge: {
    // Learn more on https://tailwindcss.com/docs/controlling-file-size/#removing-unused-css
    enabled: process.env.NODE_ENV === 'production',
    content: [
      'components/**/*.vue',
      'layouts/**/*.vue',
      'pages/**/*.vue',
      'plugins/**/*.js',
      'nuxt.config.js',
    ],
  },
}
